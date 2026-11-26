#!/usr/bin/env bash
# C11:C6 + 來源腔 48h「回放」(聯合訓練)。
#   做法:訓練清單 = 目標腔 1h × R(預設 20,約佔 30%)+ 來源腔 48h 的 MFA 時長清單;其餘與 C6 完全相同
#         (MFA 時長監督、從來源腔 48h 模型遷移初始化、大時長預測器、種子 1234、固定 40k 步、驗證集不變)。
#   想法:C9b 把時長預測器在 48h 預訓練再遷移,到 1h 微調後驗證 loss 又回到 1h 的水準(48h 學到的東西被忘掉)。
#         聯合訓練讓時長預測器一直看得到 48h 的資料,目標腔 1h 只佔一部分,同時也降低對 431 句的死背。
#   比較:C11 vs C6(種子 1234),各自在驗證集選取整;波形評估 + 預測器本身 + 命中率 + loss。
#   nohup bash c11_run.sh > ~/hakka_tts/runs/c11.log 2>&1 &
#   REPLAY_R=40 nohup bash c11_run.sh ...     # 想改目標腔比例
# 用 GPU:每個腔調訓練約 1–1.5 小時 + 評估約 20 分鐘,共約 3–4 小時。完成的步驟會自動跳過。
set -uo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
H=${HAKKA_ROOT:-$HOME/hakka_tts}; M=${MATCHA_ROOT:-$HOME/Matcha-TTS}; E=$H/exp_c/eval; C3=$H/exp_c/c3; O=$H/exp_c/h3
R=${REPLAY_R:-20}
DP_OVR="model.encoder.duration_predictor_params.kernel_size=5 model.encoder.duration_predictor_params.filter_channels_dp=512"
mkdir -p "$O"
source ~/miniconda3/etc/profile.d/conda.sh
conda activate tts
has_step40k() { compgen -G "$M/logs/train/$1/runs/*/checkpoints/step_0039999.ckpt" >/dev/null || \
                compgen -G "$M/logs/train/$1/runs/*/checkpoints/step_0040000.ckpt" >/dev/null; }
other() { [ "$1" = sixian ] && echo hailu || echo sixian; }

choose_and_eval() {  # 代號 腔 實驗 → 評估資料夾(依驗證集選的取整)
  local k=$1 d=$2 exp=$3 val name pick dir
  dir=${k}_${d}_step40000
  [ -f "$E/$dir/syllables_wave.csv" ] || { bash "$HERE/d1_pipeline.sh" "$d" "$dir" "$exp" --which step40000 \
      --data-yaml "$M/configs/data/fair1h_joint_$d.yaml" >&2 || return 1; conda activate tts; }
  val=$(python -c "import yaml;print(yaml.safe_load(open('$M/configs/data/fair1h_joint_${d}_mfadur.yaml'))['valid_filelist_path'])")
  [[ $val = /* ]] || val=$M/$val
  name=c6val_${k}_${d}
  [ -f "$C3/$name/recs.pkl" ] || python "$HERE/c3_target_check.py" --exp "$exp" --dialect "$d" --which step40000 \
      --filelist "$val" --name "$name" > "$H/runs/c11_${name}.log" 2>&1 || return 1
  pick=$(python "$HERE/c6_pick_rounding.py" --val "$C3/$name/recs.pkl" | tail -1)
  if [ "$pick" = cum ]; then
    dir=${dir}_cum
    [ -f "$E/$dir/syllables_wave.csv" ] || { bash "$HERE/d1_pipeline.sh" "$d" "$dir" "$exp" --which step40000 --rounding cum \
        --data-yaml "$M/configs/data/fair1h_joint_$d.yaml" >&2 || return 1; conda activate tts; }
  fi
  echo "  [$k $d] 取整 $pick → $dir" >&2
  echo "$dir"
}

for d in sixian hailu; do
  s=$(other "$d")
  echo "===== $d(來源 $s,目標腔重複 ${R} 次)($(date '+%m-%d %H:%M'))"
  base=c2_mfadur_fair1h_joint_${d}_transfer_s1234
  tag=fairmix${R}
  expm=c2_mfadur_${tag}_joint_${d}_transfer_s1234
  run=c11_mix${R}_dpbig_joint_${d}_s1234

  [ -f "$M/data/hakka/joint/$s/ladder_48h_mfadur.txt" ] || python "$HERE/c2_mfa2durations.py" --dialect "$s" --lists ladder_48h \
      --offset-mode margin --margin-ms 50 > "$H/runs/c11_mfadur_$s.log" 2>&1 || { echo "★ $s 48h MFA 時長失敗(看 $H/runs/c11_mfadur_$s.log)"; continue; }
  python "$HERE/c11_mix.py" --dialect "$d" --source "$s" --repeat "$R" > "$H/runs/c11_mix_$d.log" 2>&1 \
      || { echo "★ 混合清單失敗:"; cat "$H/runs/c11_mix_$d.log"; continue; }
  grep -v "^NAME=" "$H/runs/c11_mix_$d.log"
  mix=$(grep "^NAME=" "$H/runs/c11_mix_$d.log" | cut -d= -f2)
  if [ ! -f "$M/configs/experiment/$expm.yaml" ]; then
    python "$HERE/c4_size_setup.py" --base-exp "$base" --train-list "$mix" --tag "$tag" > "$H/runs/c11_setup_$d.log" 2>&1 \
        || { echo "★ 設定失敗(看 $H/runs/c11_setup_$d.log)"; continue; }
    grep -E "訓練清單|已產生" "$H/runs/c11_setup_$d.log"
  fi

  if ! has_step40k "$run"; then
    echo "----- $run 設定檢查(要看到:訓練清單是 mix 清單、驗證集不變、load_durations true、kernel_size 5、transfer_from 是來源腔 48h)"
    (cd "$M" && python matcha/train.py experiment="$expm" run_name="$run" $DP_OVR --cfg job 2>&1 \
        | grep -n -E "train_filelist_path|valid_filelist_path|load_durations|kernel_size|filter_channels_dp|transfer_from|max_steps" | head -14)
    echo "===== 訓練 $run($(date '+%m-%d %H:%M'))"
    (cd "$M" && python matcha/train.py experiment="$expm" run_name="$run" $DP_OVR) > "$H/runs/$run.log" 2>&1
    grep -E "^\[transfer\]" "$H/runs/$run.log" | head -1
    has_step40k "$run" || { echo "★ 訓練失敗:$run(看 $H/runs/$run.log)"; continue; }
  fi

  echo "===== $d:評估($(date '+%m-%d %H:%M'))"
  dir=$(choose_and_eval c11 "$d" "$run") || { echo "★ C11 評估失敗:$d"; continue; }
  c6dir=c6_${d}_step40000$([ "$(python "$HERE/c6_pick_rounding.py" --val "$C3/c6val_c6_$d/recs.pkl" | tail -1)" = cum ] && echo _cum)
  python "$HERE/c_compare.py" --base "$E/$c6dir" --method "$E/$dir" --syl wave > "$E/c11_${d}_vs_c6.txt" 2>&1
  echo "→ $E/c11_${d}_vs_c6.txt"
  python "$HERE/h3_probe.py" --dialect "$d" --sys "c6=$E/$c6dir" --sys "c11=$E/$dir" --vs c6 --wave --out "$O/c11_probe_$d.md" > /dev/null \
      && echo "→ $O/c11_probe_$d.md"
  python "$HERE/h2_metrics.py" hit --sys "c6=$E/$c6dir" --sys "c11=$E/$dir" --out "$O/c11_hit_$d.md" > /dev/null && echo "→ $O/c11_hit_$d.md"
  python "$HERE/h2_metrics.py" loss --exp "c6_tr_dpbig_fair1h_joint_${d}_s1234" --exp "$run" --out "$O/c11_loss_$d.md" > /dev/null \
      && echo "→ $O/c11_loss_$d.md"
done
echo "完成($(date '+%m-%d %H:%M'))"
