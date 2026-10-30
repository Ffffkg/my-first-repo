#!/usr/bin/env bash
# G2 一鍵:TAAI 實驗 B 的 1h 模型(目標腔 × joint/separate × 從零/遷移)合成 → MFA 對齊 → F0 特徵 → 去聲分析
#   nohup bash g2_run.sh > ~/hakka_tts/runs/g2.log 2>&1 &
# 需要先跑過 g1_run.sh(用到真人特徵 real_<腔>_test.csv、real_<腔>_train_full.csv)。
# 用 GPU(合成);8 個模型約 1.5–2 小時。所有模型一律取 best checkpoint(與 TAAI 論文相同的挑選規則),
# 並一律用「目標腔」data config 的 mel 統計量(手冊第 9.3 節)。
set -uo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
H=${HAKKA_ROOT:-$HOME/hakka_tts}
E=$H/exp_c/eval; G=$H/exp_c/g1
CFG=${MATCHA_ROOT:-$HOME/Matcha-TTS}/configs/data
WHICH=${WHICH:-best}
TRAIN_SPLIT=${TRAIN_SPLIT:-train_full}
source ~/miniconda3/etc/profile.d/conda.sh

for d in sixian hailu; do
  for rep in joint separate; do
    for kind in scratch transfer; do
      name=taai_${d}_${rep}_${kind}
      if [ ! -f "$E/$name/syllables_wave.csv" ]; then
        bash "$HERE/d1_pipeline.sh" "$d" "$name" "fair1h_${rep}_${d}_${kind}_s1234" \
             --which "$WHICH" --mode "$rep" --data-yaml "$CFG/fair1h_${rep}_${d}.yaml" --selfcheck \
          || { echo "★ $name 失敗,略過"; continue; }
      fi
      conda activate tts
      [ -f "$E/$name/syl_f0.csv" ] || python "$HERE/g1_tone_feats.py" synth --dialect "$d" --eval-dir "$E/$name"
    done
  done
  conda activate tts
  pair_args=(); pred_args=(); sys_args=(); cmp_args=()
  for rep in joint separate; do
    sc=$E/taai_${d}_${rep}_scratch/syl_f0.csv; tr=$E/taai_${d}_${rep}_transfer/syl_f0.csv
    [ -f "$sc" ] && [ -f "$tr" ] || continue
    pair_args+=(--pair "$rep" "$sc" "$tr")
    sys_args+=(--sys "${rep}_scratch=$sc" --sys "${rep}_transfer=$tr")
    cmp_args+=(--compare "${rep}_scratch" "${rep}_transfer")
    pred_args+=(--pred "$rep" "$H/exp_c/g2/g1pred_$d/pred_${rep}_scratch.csv" "$H/exp_c/g2/g1pred_$d/pred_${rep}_transfer.csv")
  done
  [ ${#pair_args[@]} -gt 0 ] || { echo "★ $d 沒有可比較的模型"; continue; }
  # G1 分類器的預測(另外存,不覆蓋 G1 主結果),給海陸 11↔33 混淆用
  python "$HERE/g1_tone_eval.py" --dialect "$d" --train "$G/real_${d}_${TRAIN_SPLIT}.csv" \
      "${sys_args[@]}" "${cmp_args[@]}" \
      --out-dir "$H/exp_c/g2/g1pred_$d"
  extra=(); [ "$d" = hailu ] && extra=("${pred_args[@]}")
  python "$HERE/g2_qusheng.py" --dialect "$d" --real "$G/real_${d}_test.csv" "${pair_args[@]}" "${extra[@]}"
done
