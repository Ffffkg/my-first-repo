#!/usr/bin/env bash
# H3 一鍵(CPU 為主,約 15–30 分鐘):
#   A. 時長 loss 的質量分布(loss 大幅降低,到底降的是哪些符號)
#   B. 預測器本身的時長準確度(不經合成與 MFA 重新對齊)+ 三個種子集成的增益
#   C. 跨腔資料對文字特徵模型的幫助 + 神經預測器與文字特徵模型的混合
#   D. (RUN_ENS=1,預設)時長集成的波形驗證(用 GPU 合成,約 30–40 分鐘)
#   bash h3_run.sh 2>&1 | tee ~/hakka_tts/runs/h3.log
set -uo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
H=${HAKKA_ROOT:-$HOME/hakka_tts}; M=${MATCHA_ROOT:-$HOME/Matcha-TTS}; E=$H/exp_c/eval; C3=$H/exp_c/c3; O=$H/exp_c/h3
SEEDS="1234 123 2026"
mkdir -p "$O"
source ~/miniconda3/etc/profile.d/conda.sh
conda activate tts
sfx() { [ "$1" = 1234 ] && echo "" || echo "_s$1"; }
other() { [ "$1" = sixian ] && echo hailu || echo sixian; }
dirs_of() {  # 系統 腔 → 三個種子的評估資料夾(依驗證集選的取整)
  local out="" s dir pick
  for s in $SEEDS; do
    dir=$1_$2$(sfx "$s")_step40000
    pick=$(python "$HERE/c6_pick_rounding.py" --val "$C3/c6val_$1_$2$(sfx "$s")/recs.pkl" 2>/dev/null | tail -1)
    [ "$pick" = cum ] && dir=${dir}_cum
    [ -f "$E/$dir/recs.pkl" ] && out+="$E/$dir,"
  done
  echo "${out%,}"
}

for d in sixian hailu; do
  s=$(other "$d")
  echo; echo "################ $d ################"
  # A. loss 質量分布
  args=()
  for pair in "1h base:c6val_base_$d" "1h C2:c6val_c2_$d" "1h C6:c6val_c6_$d" "10h base:c10val_base10_$d" "10h C2:c10val_c2_10_$d" "10h C6:c10val_c6_10_$d"; do
    lab=${pair%%:*}; nm=${pair#*:}
    [ -f "$C3/$nm/recs.pkl" ] && args+=(--recs "$lab=$C3/$nm/recs.pkl")
  done
  python "$HERE/h3_lossmap.py" "${args[@]}" --out "$O/lossmap_$d.md" > /dev/null || echo "★ loss 分布失敗"
  # B. 預測器本身 + 集成
  args=()
  for k in base c2 c5 c6; do x=$(dirs_of "$k" "$d"); [ -n "$x" ] && args+=(--sys "$k=$x"); done
  python "$HERE/h3_probe.py" --dialect "$d" "${args[@]}" --vs base --wave --out "$O/probe_$d.md" > /dev/null || echo "★ 預測器探針失敗"
  # C. 跨腔 + 混合
  valid=$(python -c "import yaml,pathlib;print(pathlib.Path(yaml.safe_load(open('$M/configs/data/fair1h_joint_$d.yaml'))['valid_filelist_path']).stem)")
  vals=""; for sd in $SEEDS; do [ -f "$C3/c6val_c6_$d$(sfx "$sd")/recs.pkl" ] && vals+="$C3/c6val_c6_$d$(sfx "$sd")/recs.pkl,"; done
  python "$HERE/h3_xdialect.py" --dialect "$d" --source "$s" --valid-list "$valid" --c6val "${vals%,}" --c6test "$(dirs_of c6 "$d")" \
      --out "$O/xdialect_$d.md" > "$H/runs/h3_xdialect_$d.log" 2>&1 || echo "★ 跨腔探針失敗(看 $H/runs/h3_xdialect_$d.log)"
  for f in lossmap probe xdialect; do echo; echo "##### $f $d"; cat "$O/${f}_$d.md" 2>/dev/null; done
done
if [ "${RUN_ENS:-1}" = 1 ]; then bash "$HERE/c12_ens.sh"; fi
echo; echo "完成($(date '+%m-%d %H:%M'))"
