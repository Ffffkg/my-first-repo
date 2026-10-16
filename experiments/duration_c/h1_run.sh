#!/usr/bin/env bash
# H1 一鍵:時長 ρ 的上限診斷(oracle-rate、純文字特徵 GBM、48h 大資料模型)。
#   nohup bash h1_run.sh > ~/hakka_tts/runs/h1.log 2>&1 &
# 用 GPU 的只有 48h 模型的合成評估(每腔約 10–15 分鐘);其他只用 CPU。
# 各系統的取整沿用 c6_pick_rounding.py 在驗證集選的結果(與 c7/c8 相同)。
set -uo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
H=${HAKKA_ROOT:-$HOME/hakka_tts}; E=$H/exp_c/eval; C3=$H/exp_c/c3
SEEDS="1234 123 2026"
source ~/miniconda3/etc/profile.d/conda.sh
conda activate tts
sfx() { [ "$1" = 1234 ] && echo "" || echo "_s$1"; }
dirs_of() {  # 系統 腔 → 三個種子的評估資料夾(依驗證集選的取整)
  local out="" s dir pick
  for s in $SEEDS; do
    dir=$1_$2$(sfx "$s")_step40000
    pick=$(python "$HERE/c6_pick_rounding.py" --val "$C3/c6val_$1_$2$(sfx "$s")/recs.pkl" 2>/dev/null | tail -1)
    [ "$pick" = cum ] && dir=${dir}_cum
    [ -f "$E/$dir/syllables_wave.csv" ] && out+="$E/$dir,"
  done
  echo "${out%,}"
}

for d in sixian hailu; do
  echo "===== $d($(date '+%m-%d %H:%M'))"
  big=hakka_joint_${d}_ladder_48h
  if [ ! -f "$E/big48h_$d/syllables_wave.csv" ]; then
    bash "$HERE/d1_pipeline.sh" "$d" "big48h_$d" "$big" --which best || echo "★ 48h 模型評估失敗:$big"
    conda activate tts
  fi
  args=()
  for k in base c2 c6; do x=$(dirs_of "$k" "$d"); [ -n "$x" ] && args+=(--sys "$k=$x"); done
  [ -f "$E/big48h_$d/syllables_wave.csv" ] && args+=(--sys "48h 模型(baseline 訓練法)=$E/big48h_$d")
  python "$HERE/h1_ceiling.py" --dialect "$d" "${args[@]}" || echo "★ H1 失敗:$d"
done
echo "完成($(date '+%m-%d %H:%M'))"
