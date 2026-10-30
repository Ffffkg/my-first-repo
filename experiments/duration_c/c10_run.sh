#!/usr/bin/env bash
# C10:C6 的做法(MFA 時長監督 + 跨腔遷移 + 大時長預測器)用在 10h 目標腔資料上,看 loss 與成功率能降 / 升多少。
# 與 C4(10h baseline、10h C2)同樣的訓練清單(ladder_10h)、驗證集、種子 1234、固定 40k 步;
# base / C2 / C6 都在驗證集選取整(與 1h 相同規則),再互相比較,並和 1h 的 C6、48h 模型並列。
#   nohup bash c10_run.sh > ~/hakka_tts/runs/c10.log 2>&1 &
# 用 GPU:訓練 2 次(約 1.5–2 小時/次)+ 評估,約 5–6 小時。完成的步驟會自動跳過。
set -uo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
H=${HAKKA_ROOT:-$HOME/hakka_tts}; M=${MATCHA_ROOT:-$HOME/Matcha-TTS}; E=$H/exp_c/eval; C3=$H/exp_c/c3; O=$H/exp_c/h2
DP_OVR="model.encoder.duration_predictor_params.kernel_size=5 model.encoder.duration_predictor_params.filter_channels_dp=512"
mkdir -p "$O"
source ~/miniconda3/etc/profile.d/conda.sh
conda activate tts
has_step40k() { compgen -G "$M/logs/train/$1/runs/*/checkpoints/step_0039999.ckpt" >/dev/null || \
                compgen -G "$M/logs/train/$1/runs/*/checkpoints/step_0040000.ckpt" >/dev/null; }

choose() {  # 代號 腔 實驗 評估名(ceil 版) [額外參數] → 依驗證集選的評估資料夾
  local k=$1 d=$2 exp=$3 dir=$4 extra=${5:-} val name pick
  has_step40k "$exp" || { echo "★ 沒有第 40000 步:$exp" >&2; return 1; }
  [ -f "$E/$dir/syllables_wave.csv" ] || { bash "$HERE/d1_pipeline.sh" "$d" "$dir" "$exp" --which step40000 $extra >&2 \
      || return 1; conda activate tts; }
  val=$(python -c "import yaml;print(yaml.safe_load(open('$M/configs/data/fair10h_joint_${d}_mfadur.yaml'))['valid_filelist_path'])")
  [[ $val = /* ]] || val=$M/$val
  name=c10val_${k}_${d}
  [ -f "$C3/$name/recs.pkl" ] || python "$HERE/c3_target_check.py" --exp "$exp" --dialect "$d" --which step40000 \
      --filelist "$val" --name "$name" > "$H/runs/c10_${name}.log" 2>&1 || return 1
  pick=$(python "$HERE/c6_pick_rounding.py" --val "$C3/$name/recs.pkl" | tail -1)
  if [ "$pick" = cum ]; then
    dir=${dir}_cum
    [ -f "$E/$dir/syllables_wave.csv" ] || { bash "$HERE/d1_pipeline.sh" "$d" "$dir" "$exp" --which step40000 --rounding cum \
        $extra >&2 || return 1; conda activate tts; }
  fi
  echo "  [$k $d] 取整 $pick → $dir" >&2
  echo "$dir"
}

for d in sixian hailu; do
  echo "===== $d($(date '+%m-%d %H:%M'))"
  tr10=fair10h_joint_${d}_transfer_s1234; c5_10=c2_mfadur_$tr10; c6_10=c6_tr_dpbig_fair10h_joint_${d}_s1234
  # ---- 設定:1h 遷移設定 → 10h(只換訓練清單)→ MFA 時長監督版
  if [ ! -f "$M/configs/experiment/$tr10.yaml" ]; then
    python "$HERE/c4_size_setup.py" --base-exp "fair1h_joint_${d}_transfer_s1234" --train-list ladder_10h --tag fair10h --force \
        || { echo "★ $d 10h 遷移設定失敗"; continue; }
  fi
  [ -f "$M/configs/experiment/$c5_10.yaml" ] || python "$HERE/c2_setup.py" --base-exp "$tr10" > "$H/runs/c10_setup_$d.log" 2>&1 \
      || { echo "★ $d C2 設定失敗(看 $H/runs/c10_setup_$d.log)"; continue; }
  # ---- 訓練 C6@10h
  if ! has_step40k "$c6_10"; then
    echo "----- $c6_10 設定檢查"
    (cd "$M" && python matcha/train.py experiment="$c5_10" run_name="$c6_10" $DP_OVR --cfg job 2>&1 \
        | grep -n -E "train_filelist_path|load_durations|kernel_size|filter_channels_dp|transfer_from|max_steps" | head -10)
    echo "===== 訓練 $c6_10($(date '+%m-%d %H:%M'))"
    (cd "$M" && python matcha/train.py experiment="$c5_10" run_name="$c6_10" $DP_OVR) > "$H/runs/$c6_10.log" 2>&1
    grep -E "^\[transfer\]" "$H/runs/$c6_10.log" | head -1
    has_step40k "$c6_10" || { echo "★ 訓練失敗:$c6_10(看 $H/runs/$c6_10.log)"; continue; }
  fi
  # ---- 評估:三個 10h 系統都在驗證集選取整
  yaml="--data-yaml $M/configs/data/fair1h_joint_$d.yaml"
  b=$(choose base10 "$d" "fair10h_joint_${d}_scratch_s1234" "c4_${d}_10h_base") || b=""
  c=$(choose c2_10 "$d" "c2_mfadur_fair10h_joint_${d}_scratch_s1234" "c4_${d}_10h_c2") || c=""
  x=$(choose c6_10 "$d" "$c6_10" "c10_${d}_10h_c6" "$yaml") || x=""
  c6_1h=c6_${d}_step40000$([ "$(python "$HERE/c6_pick_rounding.py" --val "$C3/c6val_c6_$d/recs.pkl" | tail -1)" = cum ] && echo _cum)
  [ -n "$x" ] || { echo "★ $d C6@10h 評估失敗"; continue; }
  for pair in "base10:$b" "c2_10:$c" "c6_1h:$c6_1h"; do
    k=${pair%%:*}; base=${pair#*:}
    [ -n "$base" ] && [ -f "$E/$base/syllables_wave.csv" ] || continue
    python "$HERE/c_compare.py" --base "$E/$base" --method "$E/$x" --syl wave > "$E/c10_${d}_vs_$k.txt" 2>&1
    echo "→ $E/c10_${d}_vs_$k.txt"
  done
  hit=(--sys "c6_1h=$E/$c6_1h")
  [ -n "$b" ] && hit+=(--sys "base_10h=$E/$b"); [ -n "$c" ] && hit+=(--sys "c2_10h=$E/$c")
  hit+=(--sys "c6_10h=$E/$x")
  [ -f "$E/big48h_$d/syllables_wave.csv" ] && hit+=(--sys "48h=$E/big48h_$d")
  python "$HERE/h2_metrics.py" hit "${hit[@]}" --out "$O/c10_hit_$d.md" > /dev/null && echo "→ $O/c10_hit_$d.md"
  python "$HERE/h2_metrics.py" loss --exp "c6_tr_dpbig_fair1h_joint_${d}_s1234" --exp "c2_mfadur_fair10h_joint_${d}_scratch_s1234" \
      --exp "$c6_10" --out "$O/c10_loss_$d.md" > /dev/null && echo "→ $O/c10_loss_$d.md"
done
echo "完成($(date '+%m-%d %H:%M'))"
