#!/usr/bin/env bash
# V1:XAD 換到 VITS 架構——跨腔初始化(X)在 VITS 上有沒有用?X + AD + 放大 DP(XAD)呢?
#   vsrc  = 來源腔 48 h 的 VITS(官方 MAS、DP 256/3;與 Matcha 的 48 h 來源模型同一份清單 hakka_joint_<來源>_ladder_48h)
#   vxad  = 目標腔 1 h:由 vsrc 初始化 + MFA 時長監督 + DP 5/512(= Matcha 的 c6)
#   vx    = 目標腔 1 h:由 vsrc 初始化,其餘同 VITS base(MAS、DP 256/3;= Matcha 的 xmas)
#   目標腔 1 h 的系統都是 3 種子、固定 STEPS 步(與 V0 的 vbase / vad 相同:20 萬步),驗證集選取整,
#   評估與論文主表同一套,並與 Matcha 的 base / c2 / c6、V0 的 vbase / vad 並列。
#
#   nohup bash v1_vits_xad.sh > ~/hakka_tts/runs/v1.log 2>&1 &
#   DIALECTS=sixian(預設,目標腔;來源自動取另一腔)  SRC_STEPS=300000  STEPS=200000  KINDS="vxad vx"
#   REPORT_ONLY=1 bash v1_vits_xad.sh     # 不訓練,用已完成的模型出報告(訓練途中也可以先看 vxad)
# 用 GPU:來源 48 h 約 12 小時;之後每個 1 h 模型約 8 小時(vxad 3 個、vx 3 個),合計約 2.5 天。
set -uo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
H=${HAKKA_ROOT:-$HOME/hakka_tts}; M=${MATCHA_ROOT:-$HOME/Matcha-TTS}
E=$H/exp_c/eval; C3=$H/exp_c/c3; O=$H/exp_c/v1
DIALECTS=${DIALECTS:-sixian}; SRC_STEPS=${SRC_STEPS:-300000}; STEPS=${STEPS:-200000}; KINDS=${KINDS:-"vxad vx"}
SEEDS="1234 123 2026"
DP_BIG="--dp-kernel 5 --dp-filter 512"
mkdir -p "$O" "$H/runs"
source ~/miniconda3/etc/profile.d/conda.sh
act() { set +u; conda activate "$1"; set -u; }
act tts
[ -f "$H/exp_c/v0/smoke_ok" ] || { echo "★ 先跑完 v0_vits_pilot.sh(煙霧測試與 vbase / vad)"; exit 1; }
sfx() { [ "$1" = 1234 ] && echo "" || echo "_s$1"; }
src_of() { [ "$1" = sixian ] && echo hailu || echo sixian; }
list_of() {  # yaml 鍵 → 絕對路徑
  python - "$1" "$2" "$M" <<'PY'
import sys, yaml
from pathlib import Path
v = Path(yaml.safe_load(open(sys.argv[1]))[sys.argv[2]])
print(v if v.is_absolute() else Path(sys.argv[3]) / v)
PY
}
exp_of() {
  case $1 in
    base)  echo fair1h_joint_$2_scratch_s$3 ;;
    c2)    echo c2_mfadur_fair1h_joint_$2_scratch_s$3 ;;
    c6)    echo c6_tr_dpbig_fair1h_joint_$2_s$3 ;;
    vbase) echo vits_base_fair1h_$2_s$3 ;;
    vad)   echo vits_ad_fair1h_$2_s$3 ;;
    vx)    echo vits_x_fair1h_$2_s$3 ;;
    vxad)  echo vits_xad_fair1h_$2_s$3 ;;
  esac
}
src_exp() { echo "vits_base_48h_$1_s1234"; }

# ---------------- 1. 訓練
if [ "${REPORT_ONLY:-0}" != 1 ]; then
  for d in $DIALECTS; do
    s=$(src_of "$d"); sx=$(src_exp "$s"); init=$H/vits/$sx/G_$SRC_STEPS.pth
    if [ ! -f "$init" ]; then
      tl=$(list_of "$M/configs/data/hakka_joint_${s}_ladder_48h.yaml" train_filelist_path)
      echo "===== 來源模型 $sx(清單 $(basename "$tl"),$(wc -l < "$tl") 句;$SRC_STEPS 步;$(date '+%m-%d %H:%M'))"
      python "$HERE/v_train.py" --exp "$sx" --train-list "$tl" --seed 1234 --max-steps "$SRC_STEPS" --save-every 50000 \
          > "$H/runs/$sx.log" 2>&1
      grep -E "訓練句數|★" "$H/runs/$sx.log" | head -3
      grep -E "^step" "$H/runs/$sx.log" | tail -1
      [ -f "$init" ] || { echo "★ 來源模型訓練失敗(看 $H/runs/$sx.log)"; continue; }
    fi
    for k in $KINDS; do
      for seed in $SEEDS; do
        exp=$(exp_of "$k" "$d" "$seed")
        [ -f "$H/vits/$exp/G_$STEPS.pth" ] && continue
        if [ "$k" = vxad ]; then
          tl=$(list_of "$M/configs/data/fair1h_joint_${d}_mfadur.yaml" train_filelist_path); extra="--durations mfa $DP_BIG"
        else
          tl=$(list_of "$M/configs/data/fair1h_joint_$d.yaml" train_filelist_path); extra=""
        fi
        echo "===== 訓練 $exp(由 $sx 初始化;清單 $(basename "$tl");$(date '+%m-%d %H:%M'))"
        python "$HERE/v_train.py" --exp "$exp" --train-list "$tl" $extra --init-from "$init" --seed "$seed" \
            --max-steps "$STEPS" > "$H/runs/$exp.log" 2>&1
        grep -E "訓練句數|\[transfer\]|★" "$H/runs/$exp.log" | head -4
        grep -E "^step" "$H/runs/$exp.log" | tail -1
        [ -f "$H/vits/$exp/G_$STEPS.pth" ] || echo "★ 訓練失敗:$exp(看 $H/runs/$exp.log)"
      done
    done
  done
fi

# ---------------- 2. 評估(同 v0_vits_pilot.sh:驗證集選取整後跑 D1)
val_of() { list_of "$M/configs/data/fair1h_joint_$1_mfadur.yaml" valid_filelist_path; }
choose() {  # 系統 腔 種子 → 評估資料夾名稱
  local k=$1 d=$2 s=$3 exp name pick dir arch=() extra=() which
  exp=$(exp_of "$k" "$d" "$s")
  case $k in
    v*) arch=(--arch vits); which=step$STEPS; dir=${k}_${d}$(sfx "$s")_step$STEPS
        name=c6val_${k}_${d}$(sfx "$s")_step$STEPS ;;
    *)  which=step40000; dir=${k}_${d}$(sfx "$s")_step40000; name=c6val_${k}_${d}$(sfx "$s")
        [ "$k" = c6 ] && extra=(--data-yaml "$M/configs/data/fair1h_joint_$d.yaml") ;;
  esac
  if [ ! -f "$C3/$name/recs.pkl" ]; then
    python "$HERE/c3_target_check.py" "${arch[@]}" --exp "$exp" --dialect "$d" --which "$which" \
        --filelist "$(val_of "$d")" --name "$name" > "$H/runs/c6_${name}.log" 2>&1 || { echo "★ 驗證集診斷失敗:$name" >&2; return 1; }
  fi
  pick=$(python "$HERE/c6_pick_rounding.py" --val "$C3/$name/recs.pkl" | tail -1)
  [ "$pick" = cum ] && dir=${dir}_cum
  if [ ! -f "$E/$dir/syllables_wave.csv" ]; then
    bash "$HERE/d1_pipeline.sh" "$d" "$dir" "$exp" "${arch[@]}" "${extra[@]}" --which "$which" \
        $([ "$pick" = cum ] && echo "--rounding cum") >&2 || { echo "★ 評估失敗:$dir" >&2; return 1; }
    act tts
  fi
  echo "  [$k $d s$s $which] 選 $pick → $dir" >&2
  echo "$dir"
}

for d in $DIALECTS; do
  echo "===== $d:評估($(date '+%m-%d %H:%M'))"
  sys_args=()
  for k in base c2 c6 vbase vad vx vxad; do
    out=""
    for s in $SEEDS; do
      exp=$(exp_of "$k" "$d" "$s")
      [[ $k = v* ]] && [ ! -f "$H/vits/$exp/G_$STEPS.pth" ] && { echo "  ($exp 沒有第 $STEPS 步,略過)"; continue; }
      x=$(choose "$k" "$d" "$s") && out+="$E/$x,"
    done
    [ -n "$out" ] && sys_args+=(--sys "$k=${out%,}")
  done
  [ -f "$E/big48h_$d/syllables_wave.csv" ] && sys_args+=(--sys "48h=$E/big48h_$d")
  for b in vbase vad base c6; do
    printf '%s\n' "${sys_args[@]}" | grep -q "^$b=" || continue
    echo "===== $d:時長命中率、ρ、MCD(與 $b 比較;v* = VITS,其餘 = Matcha)"
    python "$HERE/x1_overlap.py" --dialect "$d" --base "$b" "${sys_args[@]}" --out "$O/v1_${d}_vs_${b}.md" \
        | sed -n "/## 全部/,/## 排除/p" | grep -v "## 排除"
  done
done
echo "完成($(date '+%m-%d %H:%M'))"
