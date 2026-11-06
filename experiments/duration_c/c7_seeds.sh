#!/usr/bin/env bash
# C7:C6(與 C5)補訓練種子 123、2026,所有系統用驗證集選定的取整,三個種子合併比較(論文主表用)。
#   nohup bash c7_seeds.sh > ~/hakka_tts/runs/c7.log 2>&1 &
#   ONLY_C6=1 nohup bash c7_seeds.sh > ~/hakka_tts/runs/c7.log 2>&1 &     # 只補 C6(約一半時間)
# 用 GPU:先訓練 C6 4 次(2 腔 × 2 種子),再訓練 C5 4 次,每次約 1.5 小時;評估另外約 2 小時。
# 種子 1234 沿用 c6_run.sh 的結果;base、C2 的種子 123、2026 已經訓練過,只補驗證集選取整。
# 完成的步驟會自動跳過,中斷後直接重跑;合併表最後才產生,隨時可以重跑一次更新。
set -uo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
H=${HAKKA_ROOT:-$HOME/hakka_tts}; M=${MATCHA_ROOT:-$HOME/Matcha-TTS}; E=$H/exp_c/eval; C3=$H/exp_c/c3
DP_OVR="model.encoder.duration_predictor_params.kernel_size=5 model.encoder.duration_predictor_params.filter_channels_dp=512"
SEEDS="1234 123 2026"
source ~/miniconda3/etc/profile.d/conda.sh
conda activate tts
has_step40k() { compgen -G "$M/logs/train/$1/runs/*/checkpoints/step_0039999.ckpt" >/dev/null || \
                compgen -G "$M/logs/train/$1/runs/*/checkpoints/step_0040000.ckpt" >/dev/null; }
sfx() { [ "$1" = 1234 ] && echo "" || echo "_s$1"; }      # 評估資料夾名稱沿用先前的規則

exp_of() {  # 系統 腔 種子 → 實驗(run_name)
  case $1 in
    base) echo fair1h_joint_$2_scratch_s$3 ;;
    c2)   echo c2_mfadur_fair1h_joint_$2_scratch_s$3 ;;
    c5)   echo c2_mfadur_fair1h_joint_$2_transfer_s$3 ;;
    c6)   echo c6_tr_dpbig_fair1h_joint_$2_s$3 ;;
  esac
}
extra_of() { case $1 in c5|c6) echo "--data-yaml $M/configs/data/fair1h_joint_$2.yaml" ;; esac; }

train() {  # 系統 腔 種子:用種子 1234 的設定,只覆寫 seed 與 run_name(C6 另加大時長預測器)
  local k=$1 d=$2 s=$3 exp ovr=""
  exp=$(exp_of "$k" "$d" "$s")
  has_step40k "$exp" && return 0
  [ "$k" = c6 ] && ovr=$DP_OVR
  local cfg=c2_mfadur_fair1h_joint_${d}_transfer_s1234
  echo "----- $exp 設定檢查(seed: $s$([ "$k" = c6 ] && echo '、kernel_size: 5、filter_channels_dp: 512'))"
  (cd "$M" && python matcha/train.py experiment="$cfg" seed="$s" run_name="$exp" $ovr --cfg job 2>&1 \
      | grep -n -E "^seed|kernel_size|filter_channels_dp|transfer_from" | head -8)
  echo "===== 訓練 $exp($(date '+%m-%d %H:%M'))"
  (cd "$M" && python matcha/train.py experiment="$cfg" seed="$s" run_name="$exp" $ovr) > "$H/runs/$exp.log" 2>&1
  grep -E "^\[transfer\]" "$H/runs/$exp.log" | head -1
  has_step40k "$exp" || echo "★ 訓練失敗:$exp(看 $H/runs/$exp.log)"
}

choose() {  # 系統 腔 種子:驗證集選取整,需要時補 cum 評估;印出「選定的評估資料夾」
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

# ---- 1. 訓練(C6 先,C5 後)
KINDS="c6"; [ "${ONLY_C6:-0}" = 1 ] || KINDS="c6 c5"
for k in $KINDS; do
  for d in sixian hailu; do
    for s in 123 2026; do train "$k" "$d" "$s"; done
  done
done

# ---- 2. 每個系統 × 種子:驗證集選取整 → 3. 三個種子合併
for d in sixian hailu; do
  echo "===== $d:驗證集選取整($(date '+%m-%d %H:%M'))"
  declare -A DIR=()
  for s in $SEEDS; do
    for k in base c2 c5 c6; do
      DIR[$k$s]=$(choose "$k" "$d" "$s") || DIR[$k$s]=""
    done
  done
  echo "===== $d:三個種子合併"
  for pair in "base c2" "base c6" "c2 c6" "c2 c5" "c5 c6"; do
    set -- $pair
    args=(); n=0
    for s in $SEEDS; do
      [ -n "${DIR[$1$s]}" ] && [ -n "${DIR[$2$s]}" ] || continue
      args+=(--pair "$E/${DIR[$1$s]}" "$E/${DIR[$2$s]}"); n=$((n + 1))
    done
    [ $n -ge 2 ] || { echo "  ($2 vs $1:只有 $n 個種子,略過)"; continue; }
    out=$E/seeds_${d}_${2}_vs_${1}.md
    python "$HERE/c_seeds.py" --name "$d $2 vs $1($n 種子)" "${args[@]}" --out "$out" > /dev/null 2>&1 \
        && echo "→ $out" || echo "★ 合併失敗:$out"
  done
  unset DIR
done
echo "完成($(date '+%m-%d %H:%M'))"
