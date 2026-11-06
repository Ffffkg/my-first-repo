#!/usr/bin/env bash
# H4:「成功率」對「容許範圍」的曲線。base / C2 / C6(三種子合併)與 48h 模型,波形量測;
# 入聲與舒聲母音核的時長誤差落在 ±20 / 30 / 40 / 50 ms、±20% 內的比例,以及 90% 的音節誤差在幾 ms 內。只用 CPU,幾十秒。
#   bash h4_tol.sh
set -uo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
H=${HAKKA_ROOT:-$HOME/hakka_tts}; E=$H/exp_c/eval; C3=$H/exp_c/c3; O=$H/exp_c/h3
SEEDS="1234 123 2026"
mkdir -p "$O"
source ~/miniconda3/etc/profile.d/conda.sh
conda activate tts
sfx() { [ "$1" = 1234 ] && echo "" || echo "_s$1"; }
dirs_of() {
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
  python "$HERE/h2_metrics.py" hit "${args[@]}" --out "$O/tol_$d.md" | sed -n '/| 系統/,$p' | grep -v "^→"
done
