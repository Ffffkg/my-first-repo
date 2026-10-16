#!/usr/bin/env bash
# C9:針對「時長預測器過擬合」的兩個改法(H2 的 loss 表:C6 的 train dur_loss ≈ 0.002,val ≈ 0.14–0.17,差 60–80 倍)。
#   C9a = C6 + 時長預測器 dropout 0.1 → 0.5(只改這一個超參數)
#   C9b = C6 + 時長預測器先在「另一腔 48h」的 MFA 時長上預訓練,再遷移
#         (C6 的大時長預測器因為形狀不同,遷移時是從零開始學的;C9b 讓它也帶著 48h 的經驗過來)
#         來源模型:從原本的 48h 模型初始化,改用 MFA 時長監督 + 大時長預測器,再訓練 DPSRC_STEPS 步
# 都是種子 1234、固定 40k 步;各自在驗證集選取整;與 C6(種子 1234)比較。
#   nohup bash c9_run.sh > ~/hakka_tts/runs/c9.log 2>&1 &
# 用 GPU:來源預訓練 2 次(約 1 小時/次)+ 目標訓練 4 次(約 1.5 小時/次)+ 評估,共約 9–10 小時。
set -uo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
H=${HAKKA_ROOT:-$HOME/hakka_tts}; M=${MATCHA_ROOT:-$HOME/Matcha-TTS}; E=$H/exp_c/eval; C3=$H/exp_c/c3; O=$H/exp_c/h2
DP_OVR="model.encoder.duration_predictor_params.kernel_size=5 model.encoder.duration_predictor_params.filter_channels_dp=512"
DROP_OVR="model.encoder.duration_predictor_params.p_dropout=0.5"
DPSRC_STEPS=${DPSRC_STEPS:-20000}
mkdir -p "$O"
source ~/miniconda3/etc/profile.d/conda.sh
conda activate tts
has_step40k() { compgen -G "$M/logs/train/$1/runs/*/checkpoints/step_0039999.ckpt" >/dev/null || \
                compgen -G "$M/logs/train/$1/runs/*/checkpoints/step_0040000.ckpt" >/dev/null; }
other() { [ "$1" = sixian ] && echo hailu || echo sixian; }
data_lists() {  # 實驗 → 「train 清單名 valid 清單名」(不含 .txt)
  python - "$M" "$1" <<'PY'
import sys, yaml
from pathlib import Path
M, exp = Path(sys.argv[1]), sys.argv[2]
cfg = yaml.safe_load((M / "configs/experiment" / f"{exp}.yaml").read_text(encoding="utf-8"))
data = [v for it in cfg.get("defaults", []) if isinstance(it, dict) for k, v in it.items() if k.replace(" ", "") in ("override/data", "/data")][0]
d = yaml.safe_load((M / "configs/data" / f"{data}.yaml").read_text(encoding="utf-8"))
print(Path(d["train_filelist_path"]).stem, Path(d["valid_filelist_path"]).stem)
PY
}
newest_ckpt() { ls -t "$M"/logs/train/"$1"/runs/*/checkpoints/*.ckpt 2>/dev/null | head -1; }

choose_and_eval() {  # 代號 腔 實驗 → 評估資料夾(依驗證集選的取整)
  local k=$1 d=$2 exp=$3 val name pick dir
  dir=${k}_${d}_step40000
  [ -f "$E/$dir/syllables_wave.csv" ] || { bash "$HERE/d1_pipeline.sh" "$d" "$dir" "$exp" --which step40000 \
      --data-yaml "$M/configs/data/fair1h_joint_$d.yaml" >&2 || return 1; conda activate tts; }
  val=$(python -c "import yaml;print(yaml.safe_load(open('$M/configs/data/fair1h_joint_${d}_mfadur.yaml'))['valid_filelist_path'])")
  [[ $val = /* ]] || val=$M/$val
  name=c6val_${k}_${d}
  [ -f "$C3/$name/recs.pkl" ] || python "$HERE/c3_target_check.py" --exp "$exp" --dialect "$d" --which step40000 \
      --filelist "$val" --name "$name" > "$H/runs/c9_${name}.log" 2>&1 || return 1
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
  c5=c2_mfadur_fair1h_joint_${d}_transfer_s1234
  c9a=c9a_dpdrop_fair1h_joint_${d}_s1234; c9b=c9b_dpsrc_fair1h_joint_${d}_s1234
  src48=hakka_joint_${s}_ladder_48h; srcc2=c2_mfadur_joint_${s}_ladder_48h; dpsrc=dpsrc_${s}_48h

  # ---- C9b 第 1 步:來源腔 48h 的 MFA 時長 → 來源模型(大時長預測器)
  echo "===== $d:來源 $s 48h 預訓練($(date '+%m-%d %H:%M'))"
  if [ ! -f "$M/logs/train/$dpsrc/DONE" ]; then
    read -r tr va <<< "$(data_lists "$src48")"
    echo "  來源清單:$tr / $va"
    [ -f "$M/data/hakka/joint/$s/${tr}_mfadur.txt" ] && [ -f "$M/data/hakka/joint/$s/${va}_mfadur.txt" ] || \
      python "$HERE/c2_mfa2durations.py" --dialect "$s" --lists "$tr" "$va" --offset-mode margin --margin-ms 50 \
          > "$H/runs/c9_mfadur_$s.log" 2>&1 || { echo "★ $s 48h MFA 時長失敗(看 $H/runs/c9_mfadur_$s.log)"; }
    [ -f "$M/configs/experiment/$srcc2.yaml" ] || python "$HERE/c2_setup.py" --base-exp "$src48" > /dev/null \
      || echo "★ $srcc2 設定失敗"
    init=$(newest_ckpt "$src48"); best=$(ls "$M"/logs/train/"$src48"/runs/*/checkpoints/best*.ckpt 2>/dev/null | head -1)
    [ -n "$best" ] && init=$best
    echo "----- $dpsrc 設定檢查(初始化 $init)"
    (cd "$M" && python matcha/train.py experiment="$srcc2" run_name="$dpsrc" $DP_OVR ++trainer.max_steps="$DPSRC_STEPS" \
        ++transfer_from="$init" --cfg job 2>&1 | grep -n -E "kernel_size|filter_channels_dp|load_durations|max_steps|transfer_from" | head -8)
    echo "===== 訓練 $dpsrc($(date '+%m-%d %H:%M'))"
    (cd "$M" && python matcha/train.py experiment="$srcc2" run_name="$dpsrc" $DP_OVR ++trainer.max_steps="$DPSRC_STEPS" \
        ++transfer_from="$init") > "$H/runs/$dpsrc.log" 2>&1 && touch "$M/logs/train/$dpsrc/DONE" \
        || echo "★ 來源預訓練失敗(看 $H/runs/$dpsrc.log)"
    grep -E "^\[transfer\]" "$H/runs/$dpsrc.log" | head -1
  fi
  src_ckpt=$(newest_ckpt "$dpsrc")

  # ---- 訓練 C9a、C9b
  if ! has_step40k "$c9a"; then
    echo "----- $c9a 設定檢查(p_dropout: 0.5)"
    (cd "$M" && python matcha/train.py experiment="$c5" run_name="$c9a" $DP_OVR $DROP_OVR --cfg job 2>&1 \
        | grep -n -A4 "duration_predictor_params" | head -6)
    echo "===== 訓練 $c9a($(date '+%m-%d %H:%M'))"
    (cd "$M" && python matcha/train.py experiment="$c5" run_name="$c9a" $DP_OVR $DROP_OVR) > "$H/runs/$c9a.log" 2>&1
    grep -E "^\[transfer\]" "$H/runs/$c9a.log" | head -1
    has_step40k "$c9a" || echo "★ 訓練失敗:$c9a"
  fi
  if ! has_step40k "$c9b" && [ -n "$src_ckpt" ]; then
    echo "===== 訓練 $c9b(來源 $src_ckpt)($(date '+%m-%d %H:%M'))"
    (cd "$M" && python matcha/train.py experiment="$c5" run_name="$c9b" $DP_OVR transfer_from="$src_ckpt") > "$H/runs/$c9b.log" 2>&1
    grep -E "^\[transfer\]" "$H/runs/$c9b.log" | head -1      # 重置數應該回到 2 個(時長預測器這次有複製)
    has_step40k "$c9b" || echo "★ 訓練失敗:$c9b"
  fi

  # ---- 評估與比較(對 C6 種子 1234)
  echo "===== $d:評估($(date '+%m-%d %H:%M'))"
  c6dir=$(python "$HERE/c6_pick_rounding.py" --val "$C3/c6val_c6_$d/recs.pkl" | tail -1)
  c6dir=c6_${d}_step40000$([ "$c6dir" = cum ] && echo _cum)
  hit_args=(--sys "c6=$E/$c6dir")
  for pair in "c9a:$c9a" "c9b:$c9b"; do
    k=${pair%%:*}; exp=${pair#*:}
    has_step40k "$exp" || continue
    dir=$(choose_and_eval "$k" "$d" "$exp") || { echo "★ 評估失敗:$k $d"; continue; }
    python "$HERE/c_compare.py" --base "$E/$c6dir" --method "$E/$dir" --syl wave > "$E/${k}_${d}_vs_c6.txt" 2>&1
    echo "→ $E/${k}_${d}_vs_c6.txt"
    hit_args+=(--sys "$k=$E/$dir")
  done
  python "$HERE/h2_metrics.py" hit "${hit_args[@]}" --out "$O/c9_hit_$d.md" > /dev/null && echo "→ $O/c9_hit_$d.md"
  python "$HERE/h2_metrics.py" loss --exp "c6_tr_dpbig_fair1h_joint_${d}_s1234" --exp "$c9a" --exp "$c9b" \
      --out "$O/c9_loss_$d.md" > /dev/null && echo "→ $O/c9_loss_$d.md"
done
echo "完成($(date '+%m-%d %H:%M'))"
