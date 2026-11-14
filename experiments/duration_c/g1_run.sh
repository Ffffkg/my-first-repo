#!/usr/bin/env bash
# G1 一鍵:真人特徵(訓練 + 測試)→ 各合成系統特徵 → 聲調辨識率(baseline vs C2,三個種子合併)
#   bash g1_run.sh            # 兩個腔調都跑
#   bash g1_run.sh hailu      # 只跑一個
# 只用 CPU;需要 tts 環境(pyworld、scikit-learn)。合成系統要先跑過 d1_pipeline.sh(用到 wav/ 與 mfa_tg/)。
set -uo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
E=${HAKKA_ROOT:-$HOME/hakka_tts}/exp_c/eval
G=${HAKKA_ROOT:-$HOME/hakka_tts}/exp_c/g1
TRAIN_SPLIT=${TRAIN_SPLIT:-train_full}
for d in ${1:-sixian hailu}; do
  echo "===== $d ====="
  [ -f "$G/real_${d}_${TRAIN_SPLIT}.csv" ] || python "$HERE/g1_tone_feats.py" real --dialect "$d" --split "$TRAIN_SPLIT" --max-utts 3000
  [ -f "$G/real_${d}_test.csv" ] || python "$HERE/g1_tone_feats.py" real --dialect "$d" --split test
  sys_args=(); cmp_args=()
  for s in "" _s123 _s2026; do
    for m in base c2; do
      dir=$E/${m}_${d}${s}_step40000
      [ -d "$dir/mfa_tg" ] || { echo "  (略過 $dir:沒有 mfa_tg)"; continue; }
      [ -f "$dir/syl_f0.csv" ] || python "$HERE/g1_tone_feats.py" synth --dialect "$d" --eval-dir "$dir"
      sys_args+=(--sys "${m}${s:-_s1234}=$dir/syl_f0.csv")
    done
    [ -f "$E/base_${d}${s}_step40000/syl_f0.csv" ] && [ -f "$E/c2_${d}${s}_step40000/syl_f0.csv" ] && \
      cmp_args+=(--compare "base${s:-_s1234}" "c2${s:-_s1234}")
  done
  python "$HERE/g1_tone_eval.py" --dialect "$d" --train "$G/real_${d}_${TRAIN_SPLIT}.csv" \
      --ceiling "$G/real_${d}_test.csv" "${sys_args[@]}" "${cmp_args[@]}"
done
