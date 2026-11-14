#!/usr/bin/env bash
# C13:資料量 10h 補種子(123、2026),讓「資料量曲線」每個點都有統計。
#   系統:base(Matcha 原本的 MAS)、C2(MFA 時長監督)、C6(C2 + 跨腔遷移 + 大時長預測器),兩個腔調。
#   種子 1234 沿用 C4 / C10 已經做好的結果;這裡補種子 123、2026,共 12 次訓練(每次固定 40k 步)。
#   順序:先種子 123(C6 → base → C2,先四縣後海陸),再種子 2026;每個模型訓練完立刻在背景評估
#         (一次只跑一個評估:MFA 的暫存資料夾不能同時用),所以中途停掉也有完整的成果可用。
#   每個系統、每個種子各自在驗證集選取整方式(ceil / cum)。
#   最後:三個種子合併(c_seeds.py)的 C2 vs base、C6 vs base、C6 vs C2,以及命中率表。
#   另外順手把 C9 做好的「48h + MFA 時長監督 + 大時長預測器」模型也評估一次(資料量曲線的 48h 點)。
#
#   nohup bash c13_seeds10h.sh > ~/hakka_tts/runs/c13.log 2>&1 &
#   DRY=1 bash c13_seeds10h.sh        # 只印出會做的事與順序,不執行
# 用 GPU:12 次訓練,每次約 1–1.8 小時,加評估約 15–22 小時。完成的步驟會自動跳過,中斷後直接重跑。
set -uo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
H=${HAKKA_ROOT:-$HOME/hakka_tts}; M=${MATCHA_ROOT:-$HOME/Matcha-TTS}; E=$H/exp_c/eval; C3=$H/exp_c/c3; O=$H/exp_c/h3
SEEDS_NEW=${SEEDS_NEW:-"123 2026"}
DIALECTS=${DIALECTS:-"sixian hailu"}
DP_OVR="model.encoder.duration_predictor_params.kernel_size=5 model.encoder.duration_predictor_params.filter_channels_dp=512"
mkdir -p "$O" "$H/runs"

exp_of() {  # 系統 腔 種子 → 實驗名稱(run_name)
  case $1 in
    base) echo fair10h_joint_$2_scratch_s$3 ;;
    c2)   echo c2_mfadur_fair10h_joint_$2_scratch_s$3 ;;
    c6)   echo c6_tr_dpbig_fair10h_joint_$2_s$3 ;;
  esac
}
cfg_of() {  # 系統 腔 → 用來訓練的設定(種子 1234 的設定,只覆寫 seed 與 run_name)
  case $1 in
    base) echo fair10h_joint_$2_scratch_s1234 ;;
    c2)   echo c2_mfadur_fair10h_joint_$2_scratch_s1234 ;;
    c6)   echo c2_mfadur_fair10h_joint_$2_transfer_s1234 ;;
  esac
}
dir_base_of() {  # 系統 腔 種子 → 評估資料夾名稱(ceil 版);種子 1234 沿用 C4 / C10 的名稱
  if [ "$3" = 1234 ]; then
    case $1 in base) echo c4_$2_10h_base ;; c2) echo c4_$2_10h_c2 ;; c6) echo c10_$2_10h_c6 ;; esac
  else echo "c13_$2_10h_$1_s$3"; fi
}
val_name_of() {  # 系統 腔 種子 → c3_target_check 輸出名稱(驗證集的 recs.pkl)
  if [ "$3" = 1234 ]; then
    case $1 in base) echo c10val_base10_$2 ;; c2) echo c10val_c2_10_$2 ;; c6) echo c10val_c6_10_$2 ;; esac
  else echo "c13val_$1_$2_s$3"; fi
}
extra_of() { case $1 in c6) echo "--data-yaml $M/configs/data/fair1h_joint_$2.yaml" ;; esac; }   # 遷移模型用目標腔的 mel 統計量

if [ "${DRY:-0}" = 1 ]; then
  echo "DRY RUN:會依序做(訓練 → 背景評估):"
  for s in $SEEDS_NEW; do for k in c6 base c2; do for d in $DIALECTS; do
    echo "  seed $s  $k  $d  → $(exp_of "$k" "$d" "$s")  [設定 $(cfg_of "$k" "$d")$([ "$k" = c6 ] && echo ' + 大時長預測器')]  評估 $(dir_base_of "$k" "$d" "$s")"
  done; done; done
  echo "  最後:48h 點評估、三個種子合併(種子 1234 沿用 $(dir_base_of base sixian 1234)、$(dir_base_of c2 sixian 1234)、$(dir_base_of c6 sixian 1234) 等)"
  exit 0
fi

source ~/miniconda3/etc/profile.d/conda.sh
conda activate tts
# 一天的訓練最怕磁碟滿了才死:先看剩多少(每個模型的 checkpoint 與評估音檔合計約 1–2 GB)
for p in "$M" "$H"; do
  avail=$(df --output=avail -BG "$p" 2>/dev/null | tail -1 | tr -dc '0-9')
  echo "磁碟剩餘:$p → ${avail:-?} GB"
  [ -n "${avail:-}" ] && [ "$avail" -lt 40 ] && echo "★ 剩餘不到 40 GB,12 個模型可能不夠,建議先清掉不需要的舊 checkpoint"
done
has_step40k() { compgen -G "$M/logs/train/$1/runs/*/checkpoints/step_0039999.ckpt" >/dev/null || \
                compgen -G "$M/logs/train/$1/runs/*/checkpoints/step_0040000.ckpt" >/dev/null; }

train() {  # 系統 腔 種子
  local k=$1 d=$2 s=$3 exp cfg ovr=""
  exp=$(exp_of "$k" "$d" "$s"); cfg=$(cfg_of "$k" "$d")
  has_step40k "$exp" && { echo "已訓練:$exp"; return 0; }
  [ -f "$M/configs/experiment/$cfg.yaml" ] || { echo "★ 找不到設定 $cfg(先跑過 c4_run.sh / c10_run.sh)"; return 1; }
  [ "$k" = c6 ] && ovr=$DP_OVR
  echo "----- $exp 設定檢查"
  (cd "$M" && python matcha/train.py experiment="$cfg" seed="$s" run_name="$exp" $ovr --cfg job 2>&1 \
      | grep -n -E "^seed|train_filelist_path|load_durations|kernel_size|transfer_from|max_steps" | head -9)
  echo "===== 訓練 $exp($(date '+%m-%d %H:%M'))"
  (cd "$M" && python matcha/train.py experiment="$cfg" seed="$s" run_name="$exp" $ovr) > "$H/runs/$exp.log" 2>&1
  grep -E "^\[transfer\]" "$H/runs/$exp.log" | head -1
  has_step40k "$exp" || { echo "★ 訓練失敗:$exp(看 $H/runs/$exp.log)"; return 1; }
}

choose() {  # 系統 腔 種子 → 印出依驗證集選定的評估資料夾名稱(需要時先評估)
  local k=$1 d=$2 s=$3 exp dir0 vname val pick dir
  exp=$(exp_of "$k" "$d" "$s"); dir0=$(dir_base_of "$k" "$d" "$s"); vname=$(val_name_of "$k" "$d" "$s")
  has_step40k "$exp" || return 1
  [ -f "$E/$dir0/syllables_wave.csv" ] || bash "$HERE/d1_pipeline.sh" "$d" "$dir0" "$exp" --which step40000 $(extra_of "$k" "$d") >&2 \
      || { echo "★ 評估失敗:$dir0" >&2; return 1; }
  val=$(python -c "import yaml;print(yaml.safe_load(open('$M/configs/data/fair10h_joint_${d}_mfadur.yaml'))['valid_filelist_path'])")
  [[ $val = /* ]] || val=$M/$val
  [ -f "$C3/$vname/recs.pkl" ] || python "$HERE/c3_target_check.py" --exp "$exp" --dialect "$d" --which step40000 \
      --filelist "$val" --name "$vname" > "$H/runs/c13_${vname}.log" 2>&1 || { echo "★ 驗證集診斷失敗:$vname" >&2; return 1; }
  pick=$(python "$HERE/c6_pick_rounding.py" --val "$C3/$vname/recs.pkl" | tail -1)
  dir=$dir0
  if [ "$pick" = cum ]; then
    dir=${dir0}_cum
    [ -f "$E/$dir/syllables_wave.csv" ] || bash "$HERE/d1_pipeline.sh" "$d" "$dir" "$exp" --which step40000 --rounding cum \
        $(extra_of "$k" "$d") >&2 || { echo "★ 評估失敗:$dir" >&2; return 1; }
  fi
  echo "  [$k $d s$s] 取整 $pick → $dir" >&2
  echo "$dir"
}

EVAL_PID=""
queue_eval() {  # 等前一個評估做完,再在背景評估這一個
  [ -n "$EVAL_PID" ] && { wait "$EVAL_PID" 2>/dev/null || true; }
  ( choose "$@" > /dev/null ) &
  EVAL_PID=$!
}

# ---- 1. 訓練 + 背景評估
for s in $SEEDS_NEW; do
  for k in c6 base c2; do
    for d in $DIALECTS; do
      train "$k" "$d" "$s" && queue_eval "$k" "$d" "$s"
    done
  done
done
[ -n "$EVAL_PID" ] && { wait "$EVAL_PID" 2>/dev/null || true; }

# ---- 2. 48h 點:C9 做好的「48h + MFA 時長監督 + 大時長預測器」(同腔,從 48h baseline 微調 20k 步)
for d in $DIALECTS; do
  exp=dpsrc_${d}_48h; name=c13_${d}_48h_mfa
  [ -f "$M/logs/train/$exp/DONE" ] || { echo "(略過 48h 點:沒有 $exp,先跑過 c9_run.sh)"; continue; }
  which=step20000
  python -c "import sys; sys.path.insert(0, '$HERE'); from hakka_c.matcha_io import find_ckpt; find_ckpt('$exp', 'step20000')" \
      > /dev/null 2>&1 || which=last
  [ -f "$E/$name/syllables_wave.csv" ] || bash "$HERE/d1_pipeline.sh" "$d" "$name" "$exp" --which "$which" || { echo "★ 48h 評估失敗:$d"; continue; }
  conda activate tts
  vn=c13val_48h_$d
  [ -f "$C3/$vn/recs.pkl" ] || python "$HERE/c3_target_check.py" --exp "$exp" --dialect "$d" --which "$which" \
      --filelist "$M/data/hakka/joint/$d/val_mfadur.txt" --name "$vn" > "$H/runs/c13_$vn.log" 2>&1 || { echo "★ 48h 驗證集診斷失敗:$d"; continue; }
  pick=$(python "$HERE/c6_pick_rounding.py" --val "$C3/$vn/recs.pkl" | tail -1)
  dir=$name
  if [ "$pick" = cum ]; then
    dir=${name}_cum
    [ -f "$E/$dir/syllables_wave.csv" ] || bash "$HERE/d1_pipeline.sh" "$d" "$dir" "$exp" --which "$which" --rounding cum || continue
    conda activate tts
  fi
  echo "  [48h MFA $d] 取整 $pick → $dir"
  [ -f "$E/big48h_$d/syllables_wave.csv" ] && python "$HERE/c_compare.py" --base "$E/big48h_$d" --method "$E/$dir" --syl wave \
      > "$E/c13_${d}_48h_mfa_vs_base48h.txt" 2>&1 && echo "→ $E/c13_${d}_48h_mfa_vs_base48h.txt"
done

# ---- 3. 三個種子合併 + 命中率
for d in $DIALECTS; do
  echo "===== $d:合併($(date '+%m-%d %H:%M'))"
  declare -A DIR=()
  for s in 1234 $SEEDS_NEW; do for k in base c2 c6; do DIR[$k$s]=$(choose "$k" "$d" "$s" 2>/dev/null) || DIR[$k$s]=""; done; done
  for pair in "base c2" "base c6" "c2 c6"; do
    set -- $pair; args=(); n=0
    for s in 1234 $SEEDS_NEW; do
      [ -n "${DIR[$1$s]}" ] && [ -n "${DIR[$2$s]}" ] || continue
      args+=(--pair "$E/${DIR[$1$s]}" "$E/${DIR[$2$s]}"); n=$((n + 1))
    done
    [ $n -ge 2 ] || { echo "  ($2 vs $1:只有 $n 個種子,略過)"; continue; }
    out=$E/seeds10h_${d}_${2}_vs_${1}.md
    python "$HERE/c_seeds.py" --name "$d 10h $2 vs $1($n 種子)" "${args[@]}" --out "$out" > /dev/null 2>&1 \
        && echo "→ $out" || echo "★ 合併失敗:$out"
  done
  hit=()
  for k in base c2 c6; do
    x=""; for s in 1234 $SEEDS_NEW; do [ -n "${DIR[$k$s]}" ] && x+="$E/${DIR[$k$s]},"; done
    [ -n "$x" ] && hit+=(--sys "${k}_10h=${x%,}")
  done
  [ -f "$E/big48h_$d/syllables_wave.csv" ] && hit+=(--sys "48h_baseline=$E/big48h_$d")
  python "$HERE/h2_metrics.py" hit "${hit[@]}" --out "$O/hit10h_$d.md" > /dev/null && echo "→ $O/hit10h_$d.md"
  unset DIR
done
echo "完成($(date '+%m-%d %H:%M'))"
