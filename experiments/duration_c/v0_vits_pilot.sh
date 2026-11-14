#!/usr/bin/env bash
# V0:XAD 換到 VITS 架構的試跑——先看 +AD(MFA 時長監督)在 VITS 上有沒有用、整條流程通不通。
#   vbase = 官方 VITS(MAS 對齊,確定性 DP 256/3,與 Matcha 的 DP 同構),1 h 四縣,從頭訓練
#   vad   = VITS +AD(MFA 時長取代 MAS),其餘相同
#   訓練清單與 Matcha 的 base / c2 完全相同;評估(D1 命中率、ρ、MCD)與論文主表同一套程式與測試句,
#   並與 Matcha 的 base / c2(3 種子)、48h 並列,看「VITS 和 Matcha 差多少」。
#
#   nohup bash v0_vits_pilot.sh > ~/hakka_tts/runs/v0.log 2>&1 &
#   STEPS=100000(預設)  EVAL_STEPS="50000 100000"(預設,看是否已收斂)  DIALECTS=sixian(預設)
#   VSEEDS="1234 123 2026"   VITS 的訓練種子(預設只有 1234);STEPS 加大時,已完成的模型會從 last.pth 接著訓練
#   例:STEPS=200000 EVAL_STEPS=200000 VSEEDS="1234 123 2026" nohup bash v0_vits_pilot.sh >> ~/hakka_tts/runs/v0.log 2>&1 &
#   REPORT_ONLY=1 bash v0_vits_pilot.sh      # 不訓練,只出報告
# 用 GPU:第一步的煙霧測試會印出每步秒數與預估時間;1 h 資料 10 萬步約 10–15 小時(× 2 個系統)。
set -uo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
H=${HAKKA_ROOT:-$HOME/hakka_tts}; M=${MATCHA_ROOT:-$HOME/Matcha-TTS}; VR=${VITS_ROOT:-$HOME/vits}
E=$H/exp_c/eval; C3=$H/exp_c/c3; O=$H/exp_c/v0
STEPS=${STEPS:-100000}; EVAL_STEPS=${EVAL_STEPS:-"50000 $STEPS"}; DIALECTS=${DIALECTS:-sixian}
VSEEDS=${VSEEDS:-1234}; SEEDS="1234 123 2026"; VKINDS=${VKINDS:-"vbase vad"}   # VKINDS="vbase" 只訓練 VITS base
mkdir -p "$O" "$H/runs"
source ~/miniconda3/etc/profile.d/conda.sh
act() { set +u; conda activate "$1"; set -u; }
act tts
sfx() { [ "$1" = 1234 ] && echo "" || echo "_s$1"; }
list_of() {  # yaml 鍵 → 絕對路徑
  python - "$1" "$2" "$M" <<'PY'
import sys, yaml
from pathlib import Path
v = Path(yaml.safe_load(open(sys.argv[1]))[sys.argv[2]])
print(v if v.is_absolute() else Path(sys.argv[3]) / v)
PY
}

# ---------------- 0. 官方 VITS 程式與煙霧測試
if [ ! -f "$VR/models.py" ]; then
  echo "===== 下載官方 VITS($(date '+%m-%d %H:%M'))"
  git clone -q https://github.com/jaywalnut310/vits.git "$VR" && git -C "$VR" checkout -q 2e561ba || { echo "★ 下載失敗"; exit 1; }
fi
python -c "import librosa" 2>/dev/null || { echo "★ tts 環境沒有 librosa(pip install librosa)"; exit 1; }
if [ "${REPORT_ONLY:-0}" != 1 ] && [ ! -f "$O/smoke_ok" ]; then
  echo "===== 煙霧測試:30 步 +AD 訓練 + 推論一致性檢查($(date '+%m-%d %H:%M'))"
  d=${DIALECTS%% *}
  rm -rf "$H/vits/vits_smoke"
  python "$HERE/v_train.py" --exp vits_smoke --train-list "$(list_of "$M/configs/data/fair1h_joint_${d}_mfadur.yaml" train_filelist_path)" \
      --durations mfa --max-steps 30 --save-every 30 --log-every 10 --workers 2 > "$O/smoke_train.log" 2>&1
  tail -4 "$O/smoke_train.log"
  [ -f "$H/vits/vits_smoke/G_30.pth" ] || { echo "★ 煙霧測試訓練失敗,看 $O/smoke_train.log"; exit 1; }
  python "$HERE/c_eval.py" --arch vits --exp vits_smoke --which step30 --dialect "$d" --selfcheck --name vits_smoke \
      --n-boot 10 > "$O/smoke_eval.log" 2>&1
  grep -E "自我檢查" "$O/smoke_eval.log"
  grep -q "一致 ✓" "$O/smoke_eval.log" || { echo "★ 推論一致性檢查失敗,看 $O/smoke_eval.log"; exit 1; }
  rm -rf "$H/vits/vits_smoke" "$E/vits_smoke"
  touch "$O/smoke_ok"
fi

exp_of() {
  case $1 in
    base)  echo fair1h_joint_$2_scratch_s$3 ;;
    c2)    echo c2_mfadur_fair1h_joint_$2_scratch_s$3 ;;
    vbase) echo vits_base_fair1h_$2_s$3 ;;
    vad)   echo vits_ad_fair1h_$2_s$3 ;;
  esac
}

# ---------------- 1. 訓練
if [ "${REPORT_ONLY:-0}" != 1 ]; then
  for d in $DIALECTS; do
    for SEED in $VSEEDS; do for k in $VKINDS; do
      exp=$(exp_of $k "$d" $SEED)
      [ -f "$H/vits/$exp/G_$STEPS.pth" ] && continue
      if [ $k = vbase ]; then
        tl=$(list_of "$M/configs/data/fair1h_joint_$d.yaml" train_filelist_path); extra=""
      else
        tl=$(list_of "$M/configs/data/fair1h_joint_${d}_mfadur.yaml" train_filelist_path); extra="--durations mfa"
      fi
      echo "===== 訓練 $exp(清單 $(basename "$tl"),$(wc -l < "$tl") 句;$(date '+%m-%d %H:%M'))"
      python "$HERE/v_train.py" --exp "$exp" --train-list "$tl" $extra --seed $SEED --max-steps "$STEPS" \
          > "$H/runs/$exp.log" 2>&1
      grep -E "訓練句數|★" "$H/runs/$exp.log" | head -3
      grep -E "^step" "$H/runs/$exp.log" | tail -1
      [ -f "$H/vits/$exp/G_$STEPS.pth" ] || echo "★ 訓練失敗:$exp(看 $H/runs/$exp.log)"
    done; done
  done
fi

# ---------------- 2. 評估:驗證集選取整後跑 D1(與論文主表同一套)
val_of() { list_of "$M/configs/data/fair1h_joint_$1_mfadur.yaml" valid_filelist_path; }
choose() {  # 系統 腔 種子 步數 → 評估資料夾名稱
  local k=$1 d=$2 s=$3 st=$4 exp name pick dir arch=() which
  exp=$(exp_of "$k" "$d" "$s")
  case $k in vbase|vad) arch=(--arch vits); which=step$st; dir=${k}_${d}$(sfx "$s")_step$st ;;
             *) which=step40000; dir=${k}_${d}$(sfx "$s")_step40000 ;; esac
  name=c6val_${k}_${d}$(sfx "$s")$([ ${#arch[@]} -gt 0 ] && echo "_step$st")
  if [ ! -f "$C3/$name/recs.pkl" ]; then
    python "$HERE/c3_target_check.py" "${arch[@]}" --exp "$exp" --dialect "$d" --which "$which" \
        --filelist "$(val_of "$d")" --name "$name" > "$H/runs/c6_${name}.log" 2>&1 || { echo "★ 驗證集診斷失敗:$name" >&2; return 1; }
  fi
  pick=$(python "$HERE/c6_pick_rounding.py" --val "$C3/$name/recs.pkl" | tail -1)
  [ "$pick" = cum ] && dir=${dir}_cum
  if [ ! -f "$E/$dir/syllables_wave.csv" ]; then
    bash "$HERE/d1_pipeline.sh" "$d" "$dir" "$exp" "${arch[@]}" --which "$which" $([ "$pick" = cum ] && echo "--rounding cum") >&2 \
        || { echo "★ 評估失敗:$dir" >&2; return 1; }
    act tts
  fi
  echo "  [$k $d s$s $which] 選 $pick → $dir" >&2
  echo "$dir"
}

for d in $DIALECTS; do
  echo "===== $d:評估($(date '+%m-%d %H:%M'))"
  sys_args=()
  for k in base c2; do
    out=""; for s in $SEEDS; do x=$(choose $k "$d" "$s" 40000) && out+="$E/$x,"; done
    [ -n "$out" ] && sys_args+=(--sys "$k=${out%,}")
  done
  for st in $EVAL_STEPS; do
    for k in $VKINDS; do
      out=""
      for s in $VSEEDS; do
        exp=$(exp_of $k "$d" "$s")
        [ -f "$H/vits/$exp/G_$st.pth" ] || { echo "  ($exp 沒有第 $st 步,略過)"; continue; }
        x=$(choose $k "$d" "$s" "$st") && out+="$E/$x,"
      done
      [ -n "$out" ] && sys_args+=(--sys "${k}_${st}=${out%,}")
    done
  done
  [ -f "$E/big48h_$d/syllables_wave.csv" ] && sys_args+=(--sys "48h=$E/big48h_$d")
  for b in vbase_$STEPS base; do
    printf '%s\n' "${sys_args[@]}" | grep -q "^$b=" || continue
    echo "===== $d:時長命中率、ρ、MCD(與 $b 比較;VITS 種子:$VSEEDS)"
    python "$HERE/x1_overlap.py" --dialect "$d" --base "$b" "${sys_args[@]}" --out "$O/v0_${d}_vs_${b}.md" \
        | sed -n "/## 全部/,/## 排除/p" | grep -v "## 排除"
  done
done
echo "完成($(date '+%m-%d %H:%M'))"
