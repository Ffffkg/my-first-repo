#!/usr/bin/env bash
# X1 一鍵:跨腔文本重疊的敏感度分析(base / C2 / C6 三種子 + 48h),兩個腔調。只用 CPU,幾分鐘。
#   bash x1_run.sh
# 報告在 ~/hakka_tts/exp_c/x1/overlap_<腔調>.md
set -uo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
H=${HAKKA_ROOT:-$HOME/hakka_tts}; E=$H/exp_c/eval; C3=$H/exp_c/c3
SEEDS="1234 123 2026"
source ~/miniconda3/etc/profile.d/conda.sh
conda activate tts
sfx() { [ "$1" = 1234 ] && echo "" || echo "_s$1"; }
dirs_of() {   # 同 h4_tol.sh:各訓練種子的評估資料夾,ceil / cum 由驗證集選
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
  args=()
  for k in base c2 c6; do x=$(dirs_of "$k" "$d"); [ -n "$x" ] && args+=(--sys "$k=$x"); done
  [ -f "$E/big48h_$d/syllables_wave.csv" ] && args+=(--sys "48h=$E/big48h_$d")
  echo "##### $d"
  python "$HERE/x1_overlap.py" --dialect "$d" "${args[@]}"
done
