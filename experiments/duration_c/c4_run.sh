#!/usr/bin/env bash
# C4 資料量實驗一鍵:在較大的訓練資料(預設 10h)上重做 baseline vs C2(種子 1234、固定 40k 步),再用 D1 波形量測比較。
#   nohup bash c4_run.sh > ~/hakka_tts/runs/c4.log 2>&1 &
#   SIZE=5h nohup bash c4_run.sh > ~/hakka_tts/runs/c4_5h.log 2>&1 &     # 換其他資料量
# 用 GPU:4 次訓練(2 腔 × baseline/C2)約 6 小時 + 評估約 1 小時。已完成的步驟會自動跳過,中斷後可直接重跑。
set -uo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
H=${HAKKA_ROOT:-$HOME/hakka_tts}; M=${MATCHA_ROOT:-$HOME/Matcha-TTS}; E=$H/exp_c/eval
SIZE=${SIZE:-10h}; LIST=ladder_$SIZE; TAG=fair$SIZE
source ~/miniconda3/etc/profile.d/conda.sh
conda activate tts

has_step40k() { compgen -G "$M/logs/train/$1/runs/*/checkpoints/step_0039999.ckpt" >/dev/null || \
                compgen -G "$M/logs/train/$1/runs/*/checkpoints/step_0040000.ckpt" >/dev/null; }

# ---- 1. 設定(baseline 設定 → MFA 時長清單 → C2 設定)
for d in sixian hailu; do
  base=${TAG}_joint_${d}_scratch_s1234; c2=c2_mfadur_$base
  [ -f "$M/configs/experiment/$base.yaml" ] || \
    python "$HERE/c4_size_setup.py" --base-exp "fair1h_joint_${d}_scratch_s1234" --train-list "$LIST" --tag "$TAG" || { echo "★ $d 設定失敗"; continue; }
  [ -f "$M/data/hakka/joint/$d/${LIST}_mfadur.txt" ] || \
    python "$HERE/c2_mfa2durations.py" --dialect "$d" --lists "$LIST" --offset-mode margin --margin-ms 50 || { echo "★ $d MFA 時長失敗"; continue; }
  [ -f "$M/configs/experiment/$c2.yaml" ] || python "$HERE/c2_setup.py" --base-exp "$base" || { echo "★ $d C2 設定失敗"; continue; }
done

# ---- 2. 訓練
for d in sixian hailu; do
  for exp in "${TAG}_joint_${d}_scratch_s1234" "c2_mfadur_${TAG}_joint_${d}_scratch_s1234"; do
    [ -f "$M/configs/experiment/$exp.yaml" ] || continue
    if has_step40k "$exp"; then echo "已訓練:$exp"; continue; fi
    echo "===== 訓練 $exp($(date '+%m-%d %H:%M'))"
    (cd "$M" && python matcha/train.py experiment="$exp") || echo "★ 訓練失敗:$exp"
  done
done

# ---- 3. 評估(合成 + MFA + 波形比較)
for d in sixian hailu; do
  base=${TAG}_joint_${d}_scratch_s1234; c2=c2_mfadur_$base
  for pair in "base:$base" "c2:$c2"; do
    k=${pair%%:*}; exp=${pair#*:}; name=c4_${d}_${SIZE}_$k
    has_step40k "$exp" || { echo "★ 沒有第 40000 步:$exp"; continue; }
    [ -f "$E/$name/syllables_wave.csv" ] || bash "$HERE/d1_pipeline.sh" "$d" "$name" "$exp" --which step40000 || echo "★ 評估失敗:$name"
  done
  conda activate tts
  if [ -f "$E/c4_${d}_${SIZE}_base/syllables_wave.csv" ] && [ -f "$E/c4_${d}_${SIZE}_c2/syllables_wave.csv" ]; then
    python "$HERE/c_compare.py" --base "$E/c4_${d}_${SIZE}_base" --method "$E/c4_${d}_${SIZE}_c2" --syl wave > "$E/c4_${d}_${SIZE}_compare.txt" 2>&1
    python "$HERE/f1_subgroups.py" "$E/c4_${d}_${SIZE}_base/syllables_wave.csv" "$E/c4_${d}_${SIZE}_c2/syllables_wave.csv" \
        --out "$E/c4_${d}_${SIZE}_f1.md" > /dev/null 2>&1
    echo "→ $E/c4_${d}_${SIZE}_compare.txt"
  fi
done
echo "完成($(date '+%m-%d %H:%M'))"
