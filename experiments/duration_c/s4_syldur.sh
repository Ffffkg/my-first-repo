#!/usr/bin/env bash
# S4:在 XAD*(1 小時 MFA 標籤的 Matcha-XAD)上,把時長預測器換成「音節→音素雙層」,其他完全不變。
#   s4factor = 音節總長 × 音節內比例(第二層用 softmax 分配音節總長)
#   s4feat   = 預測的音節總長只當作第二層的提示特徵(Moungsri 等人的做法)
#   對照組 = x4xad(實驗一已訓練好的 XAD*):同樣的 1 小時 MFA 標籤、跨腔初始化、放大的時長預測器、三個種子、40,000 步。
#   兩種新模型都保留原本「每個符號的時長損失」,再加上音節總長的對數 L1(權重 SYL_LOSS_W),
#   s4factor 另加音節內比例的 KL(權重 SHARE_LOSS_W);權重事先固定,不依測試集調整。
#
#   先套用模型修改:bash ../matcha_patch/apply_patch.sh
#   nohup bash s4_syldur.sh > ~/hakka_tts/runs/s4.log 2>&1 &
#   VARIANTS="factor feat"(預設)  DIALECTS="sixian hailu"  REPORT_ONLY=1(只出報告)
# 用 GPU:每次訓練約 2 小時;預設 2 種模型 × 2 腔 × 3 種子 = 12 次,約 24 小時,再加評估約 6–8 小時。
set -uo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
PATCH=$HERE/../matcha_patch
H=${HAKKA_ROOT:-$HOME/hakka_tts}; M=${MATCHA_ROOT:-$HOME/Matcha-TTS}
E=$H/exp_c/eval; C3=$H/exp_c/c3; X4=$H/exp_c/x4; O=$H/exp_c/s4
DP_OVR="model.encoder.duration_predictor_params.kernel_size=5 model.encoder.duration_predictor_params.filter_channels_dp=512"
SEEDS="1234 123 2026"; VARIANTS=${VARIANTS:-"factor feat"}; DIALECTS=${DIALECTS:-"sixian hailu"}
SYL_LOSS_W=${SYL_LOSS_W:-0.5}; SHARE_LOSS_W=${SHARE_LOSS_W:-0.5}
mkdir -p "$O" "$H/runs"
source ~/miniconda3/etc/profile.d/conda.sh
act() { set +u; conda activate "$1"; set -u; }
act tts
sfx() { [ "$1" = 1234 ] && echo "" || echo "_s$1"; }
has_step40k() { compgen -G "$M/logs/train/$1/runs/*/checkpoints/step_0039999.ckpt" >/dev/null || \
                compgen -G "$M/logs/train/$1/runs/*/checkpoints/step_0040000.ckpt" >/dev/null; }

exp_of() {
  case $1 in
    base)  echo fair1h_joint_$2_scratch_s$3 ;;
    c6)    echo c6_tr_dpbig_fair1h_joint_$2_s$3 ;;
    x4xad) echo x4_xad_fair1h_joint_$2_s$3 ;;
    s4*)   echo s4_${1#s4}_fair1h_joint_$2_s$3 ;;
  esac
}
extra_of() { case $1 in c6|x4xad|s4*) echo "--data-yaml $M/configs/data/fair1h_joint_$2.yaml" ;; esac; }
val_of() {
  case $1 in
    x4xad|s4*) echo "$X4/$2/lists/val_mfadur.txt" ;;
    *) python - "$M/configs/data/fair1h_joint_$2_mfadur.yaml" "$M" <<'PY'
import sys, yaml
from pathlib import Path
v = Path(yaml.safe_load(open(sys.argv[1]))["valid_filelist_path"])
print(v if v.is_absolute() else Path(sys.argv[2]) / v)
PY
    ;;
  esac
}

# ---------------- 0. 檢查:模型修改已套用、正確;音節切分規則與 TextGrid 一致
grep -q "syl_dur" "$M/matcha/models/components/text_encoder.py" || { echo "★ 先執行:bash $PATCH/apply_patch.sh"; exit 1; }
python "$PATCH/check_syllable_duration.py" "$M" > "$O/check_model.log" 2>&1 || { echo "★ 模型檢查失敗,看 $O/check_model.log"; exit 1; }
echo "===== 模型檢查通過($(tail -1 "$O/check_model.log"))"
declare -A NASAL
for d in $DIALECTS; do
  # 音節邊界由文字(拼音)決定:用 TextGrid words 層的分組(不用時間)建查表;規則只在查不到時備用
  NASAL[$d]=$(python "$HERE/s3_syllable_check.py" --dialect "$d" --out "$O/s3_$d.md" --min-acc 0 \
      --write-table "$O/syl_table_$d.json" | tail -1)
  echo "===== $d:音節分組表"
  grep -E "^\| (onset|coda)|音節分組表" "$O/s3_$d.md"
  [ -s "$O/syl_table_$d.json" ] || { echo "★ 音節分組表建立失敗:$d"; exit 1; }
done

# ---------------- 1. 訓練
train() {
  local v=$1 d=$2 s=$3 exp X=$X4/$2 steps=${4:-}
  exp=$(exp_of "s4$v" "$d" "$s"); [ -n "$steps" ] && exp=s4_smoke_${v}_$d
  [ -z "$steps" ] && has_step40k "$exp" && return 0
  local ovr="$DP_OVR ++model.encoder.syl_dur=$v ++model.encoder.syl_nasal_rule=${NASAL[$d]} ++model.encoder.syl_table=$O/syl_table_$d.json"
  ovr+=" ++model.encoder.syl_loss_w=$SYL_LOSS_W ++model.encoder.share_loss_w=$SHARE_LOSS_W"
  local data="data.train_filelist_path=$X/lists/train_mfadur.txt data.valid_filelist_path=$X/lists/val_mfadur.txt"
  [ -n "$steps" ] && ovr+=" trainer.max_steps=$steps trainer.val_check_interval=100"
  echo "===== 訓練 $exp($(date '+%m-%d %H:%M'))"
  (cd "$M" && python matcha/train.py experiment="c2_mfadur_fair1h_joint_${d}_transfer_s1234" seed="$s" run_name="$exp" \
      $ovr $data) > "$H/runs/$exp.log" 2>&1
  grep -E "^\[transfer\]" "$H/runs/$exp.log" | head -1
  if [ -n "$steps" ]; then
    grep -q "Traceback" "$H/runs/$exp.log" && { echo "★ 試跑失敗,看 $H/runs/$exp.log"; tail -5 "$H/runs/$exp.log"; return 1; }
    return 0
  fi
  has_step40k "$exp" || echo "★ 訓練失敗:$exp(看 $H/runs/$exp.log)"
}
if [ "${REPORT_ONLY:-0}" != 1 ]; then
  if [ ! -f "$O/smoke_ok" ]; then             # 試跑 200 步(含 2 次驗證),確認訓練與驗證流程沒有錯
    for v in $VARIANTS; do train "$v" "${DIALECTS%% *}" 1234 200 || exit 1; done
    touch "$O/smoke_ok"
  fi
  for v in $VARIANTS; do for d in $DIALECTS; do for s in $SEEDS; do train "$v" "$d" "$s"; done; done; done
fi

# ---------------- 2. 評估:驗證集選取整後跑 D1(與論文主表同一套)
choose() {
  local k=$1 d=$2 s=$3 exp name pick dir
  exp=$(exp_of "$k" "$d" "$s")
  has_step40k "$exp" || { echo "★ 沒有第 40000 步:$exp" >&2; return 1; }
  name=c6val_${k}_${d}$(sfx "$s")
  [ -f "$C3/$name/recs.pkl" ] || python "$HERE/c3_target_check.py" --exp "$exp" --dialect "$d" --which step40000 \
      --filelist "$(val_of "$k" "$d")" --name "$name" > "$H/runs/c6_${name}.log" 2>&1 || { echo "★ 驗證集診斷失敗:$name" >&2; return 1; }
  pick=$(python "$HERE/c6_pick_rounding.py" --val "$C3/$name/recs.pkl" | tail -1)
  dir=${k}_${d}$(sfx "$s")_step40000$([ "$pick" = cum ] && echo _cum)
  if [ ! -f "$E/$dir/syllables_wave.csv" ]; then
    bash "$HERE/d1_pipeline.sh" "$d" "$dir" "$exp" --which step40000 $([ "$pick" = cum ] && echo "--rounding cum") \
        $(extra_of "$k" "$d") >&2 || { echo "★ 評估失敗:$dir" >&2; return 1; }
    act tts
  fi
  echo "  [$k $d s$s] 選 $pick → $dir" >&2
  echo "$dir"
}

for d in $DIALECTS; do
  echo "===== $d:評估($(date '+%m-%d %H:%M'))"
  declare -A DIRS=()
  KS="base x4xad c6"; for v in $VARIANTS; do KS+=" s4$v"; done
  for k in $KS; do
    out=""; for s in $SEEDS; do x=$(choose "$k" "$d" "$s") && out+="$E/$x,"; done
    DIRS[$k]=${out%,}
  done
  sys_args=()
  for k in $KS; do [ -n "${DIRS[$k]}" ] && sys_args+=(--sys "$k=${DIRS[$k]}"); done
  [ -f "$E/big48h_$d/syllables_wave.csv" ] && sys_args+=(--sys "48h=$E/big48h_$d")
  for b in x4xad base; do
    echo "===== $d:時長命中率、ρ、MCD(與 $b 比較)"
    python "$HERE/x1_overlap.py" --dialect "$d" --base "$b" "${sys_args[@]}" --out "$O/s4_${d}_vs_${b}.md" \
        | sed -n "/## 全部/,/## 排除/p" | grep -v "## 排除"
  done
  echo "===== $d:合成波形上的音節總長與母音誤差(測試集,三個種子合併)"
  for k in x4xad $(for v in $VARIANTS; do echo s4$v; done); do
    [ -n "${DIRS[$k]}" ] || continue
    python "$HERE/s1_syllable_diag.py" --dialect "$d" --eval-dirs "${DIRS[$k]}" --out "$O/s1test_${k}_$d.md" > /dev/null \
        && echo "  $k:" && grep -E "^\| (音節總長平均絕對誤差|母音平均絕對誤差|母音比例平均絕對誤差)" "$O/s1test_${k}_$d.md"
  done
  unset DIRS
done
echo "完成($(date '+%m-%d %H:%M'))"
