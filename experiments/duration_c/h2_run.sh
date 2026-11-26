#!/usr/bin/env bash
# H2 一鍵:先把「成功率」與 loss 的現況量清楚。只用 CPU(約 1–2 小時,大部分花在抽真人 F0)。
#   (1) 時長命中率(±20 ms / ±30 ms / ±20%):base、C2、C6(3 種子)與 48h 模型
#   (2) 訓練 / 驗證 loss:base、C2、C5、C6(種子 1234)與 48h 模型
#   (3) 聲調辨識 G1v2:同樣只看聲學,但分類器更強(前後音節完整輪廓、語速正規化、更多真人訓練句),
#       看「真人天花板」能不能拉高,合成語音是否跟著上去
#   nohup bash h2_run.sh > ~/hakka_tts/runs/h2.log 2>&1 &
set -uo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
H=${HAKKA_ROOT:-$HOME/hakka_tts}; E=$H/exp_c/eval; C3=$H/exp_c/c3; G=$H/exp_c/g1; O=$H/exp_c/h2
SEEDS="1234 123 2026"; BIG_UTTS=${BIG_UTTS:-12000}
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
  echo "===== $d($(date '+%m-%d %H:%M'))"
  # (1) 時長命中率
  args=()
  for k in base c2 c6; do x=$(dirs_of "$k" "$d"); [ -n "$x" ] && args+=(--sys "$k=$x"); done
  [ -f "$E/big48h_$d/syllables_wave.csv" ] && args+=(--sys "48h=$E/big48h_$d")
  python "$HERE/h2_metrics.py" hit "${args[@]}" --out "$O/hit_$d.md" > /dev/null && echo "→ $O/hit_$d.md" || echo "★ 命中率失敗"
  # (2) loss
  python "$HERE/h2_metrics.py" loss --exp "fair1h_joint_${d}_scratch_s1234" --exp "c2_mfadur_fair1h_joint_${d}_scratch_s1234" \
      --exp "c2_mfadur_fair1h_joint_${d}_transfer_s1234" --exp "c6_tr_dpbig_fair1h_joint_${d}_s1234" \
      --exp "hakka_joint_${d}_ladder_48h" --out "$O/loss_$d.md" > /dev/null && echo "→ $O/loss_$d.md" || echo "★ loss 失敗"
  # (3) G1v2
  big=$G/real_${d}_train_big.csv
  [ -f "$big" ] || python "$HERE/g1_tone_feats.py" real --dialect "$d" --split train_full --max-utts "$BIG_UTTS" --out "$big" \
      > "$H/runs/h2_g1feat_$d.log" 2>&1 || echo "★ 真人特徵失敗(看 $H/runs/h2_g1feat_$d.log)"
  sys_args=(); cmp_args=()
  IFS=, read -r -a A <<< "$(dirs_of base "$d")"; IFS=, read -r -a B <<< "$(dirs_of c6 "$d")"
  for i in 0 1 2; do
    [ -f "${A[$i]:-x}/syl_f0.csv" ] && [ -f "${B[$i]:-x}/syl_f0.csv" ] || continue
    sys_args+=(--sys "base_$i=${A[$i]}/syl_f0.csv" --sys "c6_$i=${B[$i]}/syl_f0.csv"); cmp_args+=(--compare "base_$i" "c6_$i")
  done
  for v in "orig3k::$G/real_${d}_train_full.csv" "rich3k:--rich:$G/real_${d}_train_full.csv" "richbig:--rich:$big"; do
    IFS=: read -r tag flag train <<< "$v"
    [ -f "$train" ] || { echo "  (略過 G1 $tag:沒有 $train)"; continue; }
    python "$HERE/g1_tone_eval.py" --dialect "$d" --train "$train" --ceiling "$G/real_${d}_test.csv" $flag \
        "${sys_args[@]}" "${cmp_args[@]}" --out-dir "$O/g1_${d}_$tag" > /dev/null 2>&1 \
        && echo "→ $O/g1_${d}_$tag/g1_summary.md" || echo "★ G1 $tag 失敗"
  done
done
echo "完成($(date '+%m-%d %H:%M'))"
