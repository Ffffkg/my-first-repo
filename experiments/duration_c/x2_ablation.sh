#!/usr/bin/env bash
# X2:XAD-TTS 的成分拆解(消融),1h、三個訓練種子、固定 40k 步,評估與論文主表相同(D1 波形量測、驗證集選取整)。
#   三個成分:X = 跨腔初始化、AD = MFA 時長監督、DP = 較大時長預測器(kernel 5、512 通道)
#     base  = 無            (已有)
#     xmas  = X             (新訓練:遷移初始化,但時長仍用 MAS)
#     c2    = AD            (已有,論文的 +AD)
#     c5    = X + AD        (已有,c7_seeds.sh 訓練過三個種子)
#     c3b   = AD + DP       (種子 1234 已有;補 123、2026)
#     c6    = X + AD + DP   (已有,論文的 XAD)
#   nohup bash x2_ablation.sh > ~/hakka_tts/runs/x2.log 2>&1 &
#   ONLY=c3b nohup bash x2_ablation.sh ...     # 只補 c3b(4 次訓練,約 8 小時)
#   REPORT_ONLY=1 bash x2_ablation.sh          # 不訓練,只用已完成的模型出報告
# 用 GPU:c3b 4 次 + xmas 6 次訓練,每次約 2 小時;每個模型評估約 30–40 分鐘。完成的步驟會自動跳過,中斷後直接重跑。
# 報告:~/hakka_tts/exp_c/x2/ablation_<腔調>_vs_<基準>.md 與 ~/hakka_tts/exp_c/x2/g1_<腔調>_<系統>_vs_base/
set -uo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
H=${HAKKA_ROOT:-$HOME/hakka_tts}; M=${MATCHA_ROOT:-$HOME/Matcha-TTS}; E=$H/exp_c/eval; C3=$H/exp_c/c3
G=$H/exp_c/g1; O=$H/exp_c/x2
DP_OVR="model.encoder.duration_predictor_params.kernel_size=5 model.encoder.duration_predictor_params.filter_channels_dp=512"
SEEDS="1234 123 2026"
TRAIN_SPLIT=${TRAIN_SPLIT:-train_full}
mkdir -p "$O" "$H/runs"
source ~/miniconda3/etc/profile.d/conda.sh
conda activate tts
has_step40k() { compgen -G "$M/logs/train/$1/runs/*/checkpoints/step_0039999.ckpt" >/dev/null || \
                compgen -G "$M/logs/train/$1/runs/*/checkpoints/step_0040000.ckpt" >/dev/null; }
sfx() { [ "$1" = 1234 ] && echo "" || echo "_s$1"; }

exp_of() {  # 系統 腔 種子 → 實驗(run_name)
  case $1 in
    base) echo fair1h_joint_$2_scratch_s$3 ;;
    xmas) echo x2_mas_fair1h_joint_$2_transfer_s$3 ;;
    c2)   echo c2_mfadur_fair1h_joint_$2_scratch_s$3 ;;
    c5)   echo c2_mfadur_fair1h_joint_$2_transfer_s$3 ;;
    c3b)  echo c3b_dpbig_fair1h_joint_$2_scratch_s$3 ;;
    c6)   echo c6_tr_dpbig_fair1h_joint_$2_s$3 ;;
  esac
}
extra_of() { case $1 in xmas|c5|c6) echo "--data-yaml $M/configs/data/fair1h_joint_$2.yaml" ;; esac; }   # 遷移模型用目標腔 mel 統計量

train() {  # 系統 腔 種子:用種子 1234 的設定,只覆寫 seed 與 run_name(c3b 另加大時長預測器)
  local k=$1 d=$2 s=$3 exp cfg ovr=""
  exp=$(exp_of "$k" "$d" "$s")
  has_step40k "$exp" && return 0
  case $k in
    c3b)  cfg=c2_mfadur_fair1h_joint_${d}_scratch_s1234; ovr=$DP_OVR ;;
    xmas) cfg=fair1h_joint_${d}_transfer_s1234 ;;
    *)    echo "★ $exp 沒有第 40000 步的模型,這個腳本不負責訓練 $k"; return 1 ;;
  esac
  echo "----- $exp 設定檢查(seed: $s$([ "$k" = c3b ] && echo '、kernel_size: 5、filter_channels_dp: 512')$([ "$k" = xmas ] && echo '、transfer_from、load_durations 應為 false'))"
  (cd "$M" && python matcha/train.py experiment="$cfg" seed="$s" run_name="$exp" $ovr --cfg job 2>&1 \
      | grep -n -E "^seed|kernel_size|filter_channels_dp|transfer_from|load_durations|max_steps" | head -9)
  echo "===== 訓練 $exp($(date '+%m-%d %H:%M'))"
  (cd "$M" && python matcha/train.py experiment="$cfg" seed="$s" run_name="$exp" $ovr) > "$H/runs/$exp.log" 2>&1
  grep -E "^\[transfer\]" "$H/runs/$exp.log" | head -1
  has_step40k "$exp" || echo "★ 訓練失敗:$exp(看 $H/runs/$exp.log)"
}

choose() {  # 系統 腔 種子:評估 + 驗證集選取整(同 c7_seeds.sh);印出選定的評估資料夾
  local k=$1 d=$2 s=$3 exp val name pick dir
  exp=$(exp_of "$k" "$d" "$s")
  has_step40k "$exp" || { echo "★ 沒有第 40000 步:$exp" >&2; return 1; }
  dir=${k}_${d}$(sfx "$s")_step40000
  if [ ! -f "$E/$dir/syllables_wave.csv" ]; then
    bash "$HERE/d1_pipeline.sh" "$d" "$dir" "$exp" --which step40000 $(extra_of "$k" "$d") >&2 || { echo "★ 評估失敗:$dir" >&2; return 1; }
    conda activate tts
  fi
  val=$(python -c "import yaml;print(yaml.safe_load(open('$M/configs/data/fair1h_joint_${d}_mfadur.yaml'))['valid_filelist_path'])")
  [[ $val = /* ]] || val=$M/$val
  name=c6val_${k}_${d}$(sfx "$s")
  [ -f "$C3/$name/recs.pkl" ] || python "$HERE/c3_target_check.py" --exp "$exp" --dialect "$d" --which step40000 \
      --filelist "$val" --name "$name" > "$H/runs/c6_${name}.log" 2>&1 || { echo "★ 驗證集診斷失敗:$name" >&2; return 1; }
  pick=$(python "$HERE/c6_pick_rounding.py" --val "$C3/$name/recs.pkl" | tail -1)
  if [ "$pick" = cum ]; then
    dir=${dir}_cum
    if [ ! -f "$E/$dir/syllables_wave.csv" ]; then
      bash "$HERE/d1_pipeline.sh" "$d" "$dir" "$exp" --which step40000 --rounding cum $(extra_of "$k" "$d") >&2 \
          || { echo "★ 評估失敗:$dir" >&2; return 1; }
      conda activate tts
    fi
  fi
  echo "  [$k $d s$s] 選 $pick → $dir" >&2
  echo "$dir"
}

# ---- 1. 訓練(c3b 先:補齊「AD + DP」;再 xmas:「只有 X」)
if [ "${REPORT_ONLY:-0}" != 1 ]; then
  for k in ${ONLY:-c3b xmas}; do
    for d in sixian hailu; do
      for s in $SEEDS; do train "$k" "$d" "$s"; done
    done
  done
fi

# ---- 2. 評估、選取整;3. 報告
for d in sixian hailu; do
  echo "===== $d:評估與驗證集選取整($(date '+%m-%d %H:%M'))"
  declare -A DIRS=()
  for k in base xmas c2 c5 c3b c6; do
    out=""
    for s in $SEEDS; do
      x=$(choose "$k" "$d" "$s") && out+="$E/$x,"
    done
    DIRS[$k]=${out%,}
    n=$(tr ',' '\n' <<< "${DIRS[$k]}" | grep -c .)
    echo "  $k:$n 個種子"
  done

  echo "===== $d:時長命中率、ρ、MCD"
  sys_args=()
  for k in base xmas c2 c5 c3b c6; do [ -n "${DIRS[$k]}" ] && sys_args+=(--sys "$k=${DIRS[$k]}"); done
  [ -f "$E/big48h_$d/syllables_wave.csv" ] && sys_args+=(--sys "48h=$E/big48h_$d")
  for b in base c2 c5; do        # 與 base 比(各成分的總效果)、與 c2 比(加在 AD 上的效果)、與 c5 比(c6 − c5 = 加大預測器)
    [ -n "${DIRS[$b]}" ] || continue
    python "$HERE/x1_overlap.py" --dialect "$d" --base "$b" "${sys_args[@]}" --out "$O/ablation_${d}_vs_${b}.md" \
        | sed -n "/## 全部/,/## 排除/p" | grep -v "## 排除"
  done

  echo "===== $d:G1 聲調辨識(與 base 配對,三種子合併)"
  if [ ! -f "$G/real_${d}_${TRAIN_SPLIT}.csv" ] || [ ! -f "$G/real_${d}_test.csv" ]; then
    echo "★ 找不到真人特徵(先跑 g1_run.sh),略過 G1"
  else
    IFS=, read -r -a A <<< "${DIRS[base]}"
    for x in "${A[@]}"; do [ -f "$x/syl_f0.csv" ] || python "$HERE/g1_tone_feats.py" synth --dialect "$d" --eval-dir "$x" > /dev/null 2>&1; done
    for k in xmas c2 c5 c3b c6; do
      IFS=, read -r -a B <<< "${DIRS[$k]}"
      [ ${#A[@]} -eq 3 ] && [ ${#B[@]} -eq 3 ] || { echo "  ($k:種子不齊,略過)"; continue; }
      for x in "${B[@]}"; do
        [ -f "$x/syl_f0.csv" ] || python "$HERE/g1_tone_feats.py" synth --dialect "$d" --eval-dir "$x" > /dev/null 2>&1 \
            || echo "★ F0 特徵失敗:$x"
      done
      sys_args=(); cmp_args=(); i=0
      for s in $SEEDS; do
        sys_args+=(--sys "base_s$s=${A[$i]}/syl_f0.csv" --sys "${k}_s$s=${B[$i]}/syl_f0.csv")
        cmp_args+=(--compare "base_s$s" "${k}_s$s"); i=$((i + 1))
      done
      gout=$O/g1_${d}_${k}_vs_base
      python "$HERE/g1_tone_eval.py" --dialect "$d" --train "$G/real_${d}_${TRAIN_SPLIT}.csv" --ceiling "$G/real_${d}_test.csv" \
          "${sys_args[@]}" "${cmp_args[@]}" --out-dir "$gout" > /dev/null 2>&1 || { echo "★ G1 失敗:$k"; continue; }
      acc=$(python - "$gout/g1_summary.md" "$k" <<'PY'
import re, sys
text = open(sys.argv[1], encoding="utf-8").read().split("## 各聲調")[0]   # 只看第一張表(全部音節)
rows = [l for l in text.splitlines() if l.startswith(f"| {sys.argv[2]}_s")]
vals = [float(re.split(r"\s*\|\s*", l)[3].split()[0]) for l in rows]
print(f"{sum(vals) / len(vals):.1f}" if vals else "?")
PY
)
      echo "  $k:聲調辨識率(三種子平均)$acc%;與 base 的差(合併):$(grep '合併' "$gout/g1_summary.md" | cut -d'|' -f3 | xargs)"
    done
  fi
  unset DIRS
done
echo "完成($(date '+%m-%d %H:%M'))"
