#!/usr/bin/env bash
# C8 一鍵:(1) C6 的整體長度校正(驗證集波形量測 → 入聲、舒聲母音核各一個係數),三個種子;
#          (2) 三種子合併:C8 vs C6 / C2 / base;
#          (3) G1 聲調辨識:C6 vs C2、C6 vs base、C8 vs C6(三種子合併);
#          (4) F1 分組偏差:base / C2 / C6 / C8(三種子)。
#   nohup bash c8_run.sh > ~/hakka_tts/runs/c8.log 2>&1 &
# 用 GPU 的只有 (1):每個種子合成驗證集 + 測試集各一次,約 2–2.5 小時;(3)(4) 只用 CPU,約 1 小時。
# 各系統的取整沿用 c6_pick_rounding.py 在驗證集選的結果。完成的步驟會自動跳過,中斷後直接重跑。
set -uo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
H=${HAKKA_ROOT:-$HOME/hakka_tts}; M=${MATCHA_ROOT:-$HOME/Matcha-TTS}; E=$H/exp_c/eval; C3=$H/exp_c/c3; G=$H/exp_c/g1
SEEDS="1234 123 2026"
TRAIN_SPLIT=${TRAIN_SPLIT:-train_full}
source ~/miniconda3/etc/profile.d/conda.sh
conda activate tts
sfx() { [ "$1" = 1234 ] && echo "" || echo "_s$1"; }
exp_of() {
  case $1 in
    base) echo fair1h_joint_$2_scratch_s$3 ;;
    c2)   echo c2_mfadur_fair1h_joint_$2_scratch_s$3 ;;
    c6)   echo c6_tr_dpbig_fair1h_joint_$2_s$3 ;;
  esac
}
extra_of() { case $1 in c6) echo "--data-yaml $M/configs/data/fair1h_joint_$2.yaml" ;; esac; }
rounding_of() {  # c7_seeds.sh 已在驗證集算好 recs.pkl;這裡只重新讀出選擇
  python "$HERE/c6_pick_rounding.py" --val "$C3/c6val_$1_$2$(sfx "$3")/recs.pkl" | tail -1
}
chosen_dir() {
  local dir=$1_$2$(sfx "$3")_step40000
  [ "$(rounding_of "$1" "$2" "$3")" = cum ] && dir=${dir}_cum
  echo "$dir"
}
dirs_of() {  # 系統 腔 → 三個種子的評估資料夾(逗號分隔,完整路徑)
  local out="" s dir
  for s in $SEEDS; do
    if [ "$1" = c8 ]; then dir=c8_$2$(sfx "$s")_step40000; else dir=$(chosen_dir "$1" "$2" "$s"); fi
    [ -f "$E/$dir/syllables_wave.csv" ] && out+="$E/$dir,"
  done
  echo "${out%,}"
}

for d in sixian hailu; do
  # ---- (1) C8:驗證集量測 → 係數 → 測試集
  echo "===== $d:C8 長度校正($(date '+%m-%d %H:%M'))"
  val=$(python -c "import yaml;print(yaml.safe_load(open('$M/configs/data/fair1h_joint_$d.yaml'))['valid_filelist_path'])")
  [[ $val = /* ]] || val=$M/$val
  for s in $SEEDS; do
    exp=$(exp_of c6 "$d" "$s"); rnd=$(rounding_of c6 "$d" "$s")
    vdir=c8val_c6_$d$(sfx "$s"); tdir=c8_$d$(sfx "$s")_step40000
    [ -n "$rnd" ] || { echo "★ 讀不到 C6 的取整選擇:$d s$s"; continue; }
    if [ ! -f "$E/$vdir/syllables_wave.csv" ]; then
      bash "$HERE/d1_pipeline.sh" "$d" "$vdir" "$exp" --which step40000 --rounding "$rnd" --filelist "$val" \
          $(extra_of c6 "$d") || { echo "★ 驗證集評估失敗:$vdir"; continue; }
      conda activate tts
    fi
    read -r s_et s_non < <(python "$HERE/c8_tempo.py" --val-dir "$E/$vdir" | tee "$H/runs/c8_${vdir}.txt" | tail -1)
    [ -n "${s_non:-}" ] || { echo "★ 係數估計失敗:$vdir"; continue; }
    sed -n '2,3p' "$H/runs/c8_${vdir}.txt" | sed "s/^/  [s$s] /"
    echo "  [s$s] 取整 $rnd、s_et = $s_et、s_non = $s_non"
    if [ ! -f "$E/$tdir/syllables_wave.csv" ]; then
      bash "$HERE/d1_pipeline.sh" "$d" "$tdir" "$exp" --which step40000 --rounding "$rnd" --s-et "$s_et" --s-non "$s_non" \
          $(extra_of c6 "$d") || { echo "★ 測試集評估失敗:$tdir"; continue; }
      conda activate tts
    fi
  done

  # ---- (2) 三種子合併
  echo "===== $d:三種子合併"
  for pair in "c6 c8" "c2 c8" "base c8"; do
    set -- $pair
    IFS=, read -r -a A <<< "$(dirs_of "$1" "$d")"; IFS=, read -r -a B <<< "$(dirs_of "$2" "$d")"
    [ ${#A[@]} -eq 3 ] && [ ${#B[@]} -eq 3 ] || { echo "  ($2 vs $1:種子不齊,略過)"; continue; }
    args=(); for i in 0 1 2; do args+=(--pair "${A[$i]}" "${B[$i]}"); done
    out=$E/seeds_${d}_${2}_vs_${1}.md
    python "$HERE/c_seeds.py" --name "$d $2 vs $1(3 種子)" "${args[@]}" --out "$out" > /dev/null 2>&1 \
        && echo "→ $out" || echo "★ 合併失敗:$out"
  done

  # ---- (3) G1 聲調辨識
  echo "===== $d:G1 聲調辨識($(date '+%m-%d %H:%M'))"
  if [ ! -f "$G/real_${d}_${TRAIN_SPLIT}.csv" ] || [ ! -f "$G/real_${d}_test.csv" ]; then
    echo "★ 找不到真人特徵(先跑 g1_run.sh),略過 G1"
  else
    for k in base c2 c6 c8; do
      IFS=, read -r -a DIRS <<< "$(dirs_of "$k" "$d")"
      for x in "${DIRS[@]}"; do
        [ -f "$x/syl_f0.csv" ] || python "$HERE/g1_tone_feats.py" synth --dialect "$d" --eval-dir "$x" > /dev/null 2>&1 \
            || echo "★ F0 特徵失敗:$x"
      done
    done
    for pair in "c2 c6" "base c6" "c6 c8"; do
      set -- $pair
      IFS=, read -r -a A <<< "$(dirs_of "$1" "$d")"; IFS=, read -r -a B <<< "$(dirs_of "$2" "$d")"
      [ ${#A[@]} -eq 3 ] && [ ${#B[@]} -eq 3 ] || { echo "  (G1 $2 vs $1:種子不齊,略過)"; continue; }
      sys_args=(); cmp_args=(); i=0
      for s in $SEEDS; do
        sys_args+=(--sys "$1_s$s=${A[$i]}/syl_f0.csv" --sys "$2_s$s=${B[$i]}/syl_f0.csv")
        cmp_args+=(--compare "$1_s$s" "$2_s$s"); i=$((i + 1))
      done
      out=$G/c8_${d}_${2}_vs_${1}
      python "$HERE/g1_tone_eval.py" --dialect "$d" --train "$G/real_${d}_${TRAIN_SPLIT}.csv" --ceiling "$G/real_${d}_test.csv" \
          "${sys_args[@]}" "${cmp_args[@]}" --out-dir "$out" > /dev/null 2>&1 \
          && echo "→ $out/g1_summary.md" || echo "★ G1 失敗:$out"
    done
  fi

  # ---- (4) F1 分組偏差(三種子)
  f1_args=()
  for k in base c2 c6 c8; do x=$(dirs_of "$k" "$d"); [ -n "$x" ] && f1_args+=(--sys "$k=$x"); done
  python "$HERE/c8_f1_seeds.py" "${f1_args[@]}" --out "$E/c8_f1_$d.md" > /dev/null 2>&1 \
      && echo "→ $E/c8_f1_$d.md" || echo "★ F1 失敗"
done
echo "完成($(date '+%m-%d %H:%M'))"
