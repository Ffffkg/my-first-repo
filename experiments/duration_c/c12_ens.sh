#!/usr/bin/env bash
# C12:時長集成的波形驗證。主模型 = C6 種子 1234(編碼器輸出與解碼器),時長 = C6 三個種子的預測取 log 平均。
# 先看 h3_probe.py 的「集成相對單一種子的增益」有沒有明顯好處,再用這個確認合成音量出來也有。
#   bash c12_ens.sh        # 用 GPU 合成,每個腔調約 15–20 分鐘(可以和訓練同時跑,會稍微拖慢訓練)
set -uo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
H=${HAKKA_ROOT:-$HOME/hakka_tts}; M=${MATCHA_ROOT:-$HOME/Matcha-TTS}; E=$H/exp_c/eval; C3=$H/exp_c/c3; O=$H/exp_c/h3
mkdir -p "$O"
source ~/miniconda3/etc/profile.d/conda.sh
conda activate tts
for d in sixian hailu; do
  prim=c6_tr_dpbig_fair1h_joint_${d}_s1234
  ens=c6_tr_dpbig_fair1h_joint_${d}_s123,c6_tr_dpbig_fair1h_joint_${d}_s2026
  rnd=$(python "$HERE/c6_pick_rounding.py" --val "$C3/c6val_c6_$d/recs.pkl" | tail -1)
  c6dir=c6_${d}_step40000$([ "$rnd" = cum ] && echo _cum)
  name=c12ens_${d}_step40000
  echo "===== $d:時長集成(取整 $rnd)($(date '+%m-%d %H:%M'))"
  [ -f "$E/$name/syllables_wave.csv" ] || bash "$HERE/d1_pipeline.sh" "$d" "$name" "$prim" --which step40000 --rounding "$rnd" \
      --data-yaml "$M/configs/data/fair1h_joint_$d.yaml" --ens-exps "$ens" || { echo "★ 集成合成失敗:$d"; continue; }
  conda activate tts
  python "$HERE/c_compare.py" --base "$E/$c6dir" --method "$E/$name" --syl wave > "$E/c12_${d}_vs_c6.txt" 2>&1 && echo "→ $E/c12_${d}_vs_c6.txt"
  python "$HERE/h2_metrics.py" hit --sys "c6_1234=$E/$c6dir" --sys "c6_3種子時長集成=$E/$name" --out "$O/c12_hit_$d.md" > /dev/null \
      && echo "→ $O/c12_hit_$d.md"
done
echo "完成($(date '+%m-%d %H:%M'))"
