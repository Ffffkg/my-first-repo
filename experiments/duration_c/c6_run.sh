#!/usr/bin/env bash
# C6:遷移初始化 + 較大時長預測器(C5 + C3b 合併),並讓每個系統在「驗證集」上選取整方式(ceil / cum)。
#   C6 = C5 的設定(遷移、MFA 時長監督)+ kernel 5、512 通道的時長預測器;
#        形狀不同的時長預測器權重不會從來源複製(遷移程式自動重置),其他部分照樣遷移。
#   取整:base、C2、C5、C3b、C6 各自用 c6_pick_rounding.py 在驗證集上選,測試集不參與;
#        最後用「各自選定的取整」互相比較。
#   nohup bash c6_run.sh > ~/hakka_tts/runs/c6.log 2>&1 &
# 用 GPU:先做評估(約 1–1.5 小時),再訓練 C6 兩次(約 3.5 小時)。完成的步驟會自動跳過,中斷後直接重跑。
set -uo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
H=${HAKKA_ROOT:-$HOME/hakka_tts}; M=${MATCHA_ROOT:-$HOME/Matcha-TTS}; E=$H/exp_c/eval; C3=$H/exp_c/c3
DP_OVR="model.encoder.duration_predictor_params.kernel_size=5 model.encoder.duration_predictor_params.filter_channels_dp=512"
source ~/miniconda3/etc/profile.d/conda.sh
conda activate tts
has_step40k() { compgen -G "$M/logs/train/$1/runs/*/checkpoints/step_0039999.ckpt" >/dev/null || \
                compgen -G "$M/logs/train/$1/runs/*/checkpoints/step_0040000.ckpt" >/dev/null; }

exp_of() {  # 系統代號 → 實驗名稱
  case $1 in
    base) echo fair1h_joint_$2_scratch_s1234 ;;
    c2)   echo c2_mfadur_fair1h_joint_$2_scratch_s1234 ;;
    c5)   echo c2_mfadur_fair1h_joint_$2_transfer_s1234 ;;
    c3b)  echo c3b_dpbig_fair1h_joint_$2_scratch_s1234 ;;
    c6)   echo c6_tr_dpbig_fair1h_joint_$2_s1234 ;;
  esac
}
extra_of() {  # 遷移模型用目標腔的 mel 統計量(與 TAAI 相同)
  case $1 in c5|c6) echo "--data-yaml $M/configs/data/fair1h_joint_$2.yaml" ;; esac
}

# 驗證集選取整 → 需要時補 cum 評估;把選到的評估資料夾名稱寫進 $E/c6_choice_<腔>.txt
choose() {
  local k=$1 d=$2 exp val name pick
  exp=$(exp_of "$k" "$d")
  has_step40k "$exp" || { echo "★ 沒有第 40000 步:$exp"; return 1; }
  val=$(python -c "import yaml;print(yaml.safe_load(open('$M/configs/data/fair1h_joint_${d}_mfadur.yaml'))['valid_filelist_path'])")
  [[ $val = /* ]] || val=$M/$val
  name=c6val_${k}_${d}
  [ -f "$C3/$name/recs.pkl" ] || python "$HERE/c3_target_check.py" --exp "$exp" --dialect "$d" --which step40000 \
      --filelist "$val" --name "$name" > "$H/runs/c6_${name}.log" 2>&1 || { echo "★ 驗證集診斷失敗:$name"; return 1; }
  pick=$(python "$HERE/c6_pick_rounding.py" --val "$C3/$name/recs.pkl" | tee "$H/runs/c6_${name}_pick.txt" | tail -1)
  sed -n '/| 取整/,$p' "$H/runs/c6_${name}_pick.txt" | head -4 | sed "s/^/  [$k $d] /"
  local dir=${k}_${d}_step40000
  if [ "$pick" = cum ]; then
    dir=${dir}_cum
    [ -f "$E/$dir/syllables_wave.csv" ] || bash "$HERE/d1_pipeline.sh" "$d" "$dir" "$exp" --which step40000 --rounding cum \
        $(extra_of "$k" "$d") || { echo "★ 評估失敗:$dir"; return 1; }
    conda activate tts
  fi
  [ -f "$E/$dir/syllables_wave.csv" ] || { echo "★ 找不到 $E/$dir"; return 1; }
  echo "$k $pick $dir" >> "$E/c6_choice_$d.txt"
}

compare_all() {
  local d=$1 f=$E/c6_choice_$d.txt
  declare -A DIR
  while read -r k pick dir; do DIR[$k]=$dir; done < "$f"
  for pair in "base c2" "c2 c5" "c2 c3b" "c2 c6" "c5 c6" "base c6"; do
    set -- $pair
    [ -n "${DIR[$1]:-}" ] && [ -n "${DIR[$2]:-}" ] || continue
    out=$E/final_${d}_${2}_vs_${1}.txt
    python "$HERE/c_compare.py" --base "$E/${DIR[$1]}" --method "$E/${DIR[$2]}" --syl wave > "$out" 2>&1
    echo "→ $out"
  done
  [ -n "${DIR[c2]:-}" ] && [ -n "${DIR[c6]:-}" ] && python "$HERE/f1_subgroups.py" \
      "$E/${DIR[c2]}/syllables_wave.csv" "$E/${DIR[c6]}/syllables_wave.csv" --out "$E/final_${d}_c6_f1_vs_c2.md" > /dev/null 2>&1
}

for d in sixian hailu; do
  echo "===== $d:驗證集選取整(base、C2、C5、C3b)($(date '+%m-%d %H:%M'))"
  rm -f "$E/c6_choice_$d.txt"
  for k in base c2 c5 c3b; do choose "$k" "$d"; done

  c5=$(exp_of c5 "$d"); c6=$(exp_of c6 "$d")
  if ! has_step40k "$c6"; then
    echo "----- $c6 設定檢查(kernel_size: 5、filter_channels_dp: 512、transfer_from)"
    (cd "$M" && python matcha/train.py experiment="$c5" run_name="$c6" $DP_OVR --cfg job 2>&1 \
        | grep -n -E "kernel_size|filter_channels_dp|transfer_from|load_durations|use_precomputed" | head -8)
    echo "===== 訓練 $c6($(date '+%m-%d %H:%M'))"
    (cd "$M" && python matcha/train.py experiment="$c5" run_name="$c6" $DP_OVR) 2>&1 \
        | tee "$H/runs/$c6.log" | grep -E "^\[transfer\]|跨腔遷移" || true
    has_step40k "$c6" || echo "★ 訓練失敗:$c6(看 $H/runs/$c6.log)"
  fi
  if has_step40k "$c6"; then
    [ -f "$E/c6_${d}_step40000/syllables_wave.csv" ] || bash "$HERE/d1_pipeline.sh" "$d" "c6_${d}_step40000" "$c6" \
        --which step40000 $(extra_of c6 "$d") || echo "★ 評估失敗:c6_${d}_step40000"
    conda activate tts
    choose c6 "$d"
  fi
  echo "===== $d:各自用選定的取整互相比較"
  compare_all "$d"
  cat "$E/c6_choice_$d.txt"
done
echo "完成($(date '+%m-%d %H:%M'))"
