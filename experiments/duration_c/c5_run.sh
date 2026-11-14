#!/usr/bin/env bash
# C5 + C3b:在 1h、種子 1234、固定 40k 步下,試兩種改良,都跟目前的 C2(從零)比較;評估一律用 D1 波形量測。
#   C5  = C2 + 跨腔遷移初始化:由 TAAI 的遷移設定(fair1h_joint_<腔>_transfer_s1234)產生 MFA 時長監督版本
#   C3b = C2 + 較大的時長預測器:kernel 3→5(看得到前後各約 2 個音素,包含後面的停頓)、通道 256→512
#   nohup bash c5_run.sh > ~/hakka_tts/runs/c5.log 2>&1 &
# 用 GPU:4 次訓練約 6 小時 + 評估。完成的步驟會自動跳過,中斷後直接重跑。
set -uo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
H=${HAKKA_ROOT:-$HOME/hakka_tts}; M=${MATCHA_ROOT:-$HOME/Matcha-TTS}; E=$H/exp_c/eval
DP_OVR="model.encoder.duration_predictor_params.kernel_size=5 model.encoder.duration_predictor_params.filter_channels_dp=512"
source ~/miniconda3/etc/profile.d/conda.sh
conda activate tts
has_step40k() { compgen -G "$M/logs/train/$1/runs/*/checkpoints/step_0039999.ckpt" >/dev/null || \
                compgen -G "$M/logs/train/$1/runs/*/checkpoints/step_0040000.ckpt" >/dev/null; }

for d in sixian hailu; do
  c2=c2_mfadur_fair1h_joint_${d}_scratch_s1234
  c5=c2_mfadur_fair1h_joint_${d}_transfer_s1234
  c3b=c3b_dpbig_fair1h_joint_${d}_scratch_s1234

  # ---- C5 設定:沿用遷移設定的一切(含初始化來源),只加 MFA 時長監督
  if [ ! -f "$M/configs/experiment/$c5.yaml" ]; then
    python "$HERE/c2_setup.py" --base-exp "fair1h_joint_${d}_transfer_s1234" || { echo "★ $d C5 設定失敗,略過此腔"; continue; }
  fi
  echo "----- $c5 設定檢查(要看到 durations 為 true,以及遷移初始化的設定)"
  (cd "$M" && python matcha/train.py experiment="$c5" --cfg job 2>&1 | grep -n -i -E "durations|filelist|transfer|init|ckpt" | head -20)
  echo "----- $c3b 設定檢查(kernel_size: 5、filter_channels_dp: 512)"
  (cd "$M" && python matcha/train.py experiment="$c2" run_name="$c3b" $DP_OVR --cfg job 2>&1 | grep -n -A4 "duration_predictor_params" | head -8)

  # ---- 訓練
  if ! has_step40k "$c5"; then
    echo "===== 訓練 $c5($(date '+%m-%d %H:%M'))"
    (cd "$M" && python matcha/train.py experiment="$c5") || echo "★ 訓練失敗:$c5"
  fi
  if ! has_step40k "$c3b"; then
    echo "===== 訓練 $c3b($(date '+%m-%d %H:%M'))"
    (cd "$M" && python matcha/train.py experiment="$c2" run_name="$c3b" $DP_OVR) || echo "★ 訓練失敗:$c3b"
  fi

  # ---- 評估(遷移模型用目標腔的 mel 統計量,與 TAAI 相同)
  for pair in "c5:$c5:--data-yaml $M/configs/data/fair1h_joint_$d.yaml" "c3b:$c3b:"; do
    IFS=: read -r k exp extra <<< "$pair"
    name=${k}_${d}_step40000
    has_step40k "$exp" || { echo "★ 沒有第 40000 步:$exp"; continue; }
    [ -f "$E/$name/syllables_wave.csv" ] || bash "$HERE/d1_pipeline.sh" "$d" "$name" "$exp" --which step40000 $extra \
      || echo "★ 評估失敗:$name"
  done
  conda activate tts
  for k in c5 c3b; do
    [ -f "$E/${k}_${d}_step40000/syllables_wave.csv" ] || continue
    python "$HERE/c_compare.py" --base "$E/c2_${d}_step40000" --method "$E/${k}_${d}_step40000" --syl wave \
        > "$E/${k}_${d}_vs_c2.txt" 2>&1
    python "$HERE/c_compare.py" --base "$E/base_${d}_step40000" --method "$E/${k}_${d}_step40000" --syl wave \
        > "$E/${k}_${d}_vs_base.txt" 2>&1
    python "$HERE/f1_subgroups.py" "$E/c2_${d}_step40000/syllables_wave.csv" "$E/${k}_${d}_step40000/syllables_wave.csv" \
        --out "$E/${k}_${d}_f1_vs_c2.md" > /dev/null 2>&1
    echo "→ $E/${k}_${d}_vs_c2.txt"
  done
  # 參考:遷移 baseline(TAAI,best ckpt,G2 已合成)vs C5
  [ -f "$E/taai_${d}_joint_transfer/syllables_wave.csv" ] && [ -f "$E/c5_${d}_step40000/syllables_wave.csv" ] && \
    python "$HERE/c_compare.py" --base "$E/taai_${d}_joint_transfer" --method "$E/c5_${d}_step40000" --syl wave \
        > "$E/c5_${d}_vs_transfer_base.txt" 2>&1
done
echo "完成($(date '+%m-%d %H:%M'))"
