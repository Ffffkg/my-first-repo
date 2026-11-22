#!/usr/bin/env bash
# S1:在「驗證集」上合成 XAD*(1 小時 MFA 標籤的 Matcha-XAD,三個種子),用 MFA 量合成語音,
#     判斷入聲母音的錯誤來自「整個音節的長度」還是「音節內部的時間分配」(s1_syllable_diag.py)。
#     不需要訓練;測試集完全不碰。取整方式沿用各模型在驗證集上已選好的設定。
#
#   bash s1_run.sh > ~/hakka_tts/runs/s1.log 2>&1
#   SYSTEMS="x4xad c6"(預設只有 x4xad;c6 = 論文版 XAD,主實驗 MFA 標籤)  DIALECTS="sixian hailu"
# 用 GPU:每個模型合成 100 句 × 5 個生成種子約 2–3 分鐘,MFA 對齊約 3–5 分鐘;預設 6 個模型,約 1 小時。
set -uo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
H=${HAKKA_ROOT:-$HOME/hakka_tts}; M=${MATCHA_ROOT:-$HOME/Matcha-TTS}
E=$H/exp_c/eval; C3=$H/exp_c/c3; O=$H/exp_c/s1; X4=$H/exp_c/x4
SYSTEMS=${SYSTEMS:-x4xad}; DIALECTS=${DIALECTS:-"sixian hailu"}; SEEDS="1234 123 2026"
mkdir -p "$O" "$H/runs"
source ~/miniconda3/etc/profile.d/conda.sh
act() { set +u; conda activate "$1"; set -u; }
act tts
sfx() { [ "$1" = 1234 ] && echo "" || echo "_s$1"; }
exp_of() { case $1 in x4xad) echo x4_xad_fair1h_joint_$2_s$3 ;; c6) echo c6_tr_dpbig_fair1h_joint_$2_s$3 ;; esac; }
val_of() {
  case $1 in
    x4xad) echo "$X4/$2/lists/val_mfadur.txt" ;;
    *) python - "$M/configs/data/fair1h_joint_$2_mfadur.yaml" "$M" <<'PY'
import sys, yaml
from pathlib import Path
v = Path(yaml.safe_load(open(sys.argv[1]))["valid_filelist_path"])
print(v if v.is_absolute() else Path(sys.argv[2]) / v)
PY
    ;;
  esac
}

for k in $SYSTEMS; do
  for d in $DIALECTS; do
    echo "===== $k $d:驗證集合成與 MFA 對齊($(date '+%m-%d %H:%M'))"
    val=$(val_of "$k" "$d")
    [ -f "$val" ] || { echo "★ 找不到驗證集清單 $val"; continue; }
    dirs=""
    for s in $SEEDS; do
      exp=$(exp_of "$k" "$d" "$s")
      rec=$C3/c6val_${k}_${d}$(sfx "$s")/recs.pkl
      [ -f "$rec" ] || { echo "★ 找不到 $rec(取整方式在驗證集上選的紀錄)"; continue; }
      pick=$(python "$HERE/c6_pick_rounding.py" --val "$rec" | tail -1)
      dir=s1val_${k}_${d}$(sfx "$s")_step40000$([ "$pick" = cum ] && echo _cum)
      if [ ! -f "$E/$dir/mfa_tg/.done" ]; then
        bash "$HERE/d1_pipeline.sh" "$d" "$dir" "$exp" --which step40000 --filelist "$val" \
            --data-yaml "$M/configs/data/fair1h_joint_$d.yaml" $([ "$pick" = cum ] && echo "--rounding cum") \
            > "$O/$dir.log" 2>&1 && touch "$E/$dir/mfa_tg/.done" || { echo "★ 合成或對齊失敗:$dir(看 $O/$dir.log)"; act tts; continue; }
        act tts
      fi
      echo "  種子 $s:取整 $pick → $dir"
      dirs+="$E/$dir,"
    done
    [ -n "$dirs" ] || continue
    python "$HERE/s1_syllable_diag.py" --dialect "$d" --eval-dirs "${dirs%,}" --out "$O/s1_${k}_${d}.md" \
        || echo "★ 分析失敗:$k $d"
  done
done
echo "完成($(date '+%m-%d %H:%M'))"
