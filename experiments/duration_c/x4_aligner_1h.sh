#!/usr/bin/env bash
# X4:目標腔「從頭到尾只用 1 h」——MFA 也只用 1 h 訓練子集訓練,再用它的時長標籤重訓 +AD 與 XAD。
#   主實驗的 MFA 用整個腔調的錄音訓練(X3 已確認改用 train_full 結論不變);X4 回答更嚴格的問題:
#   一個真的只有 1 小時錄音的新腔調,連對齊器都只能用這 1 小時,時長標籤會變差多少?XAD 還有沒有效?
#   系統(都是 1 h、三個訓練種子、固定 40k 步、驗證集選取整,評估與論文主表相同):
#     x4ad   = +AD,但時長標籤來自 1 h MFA
#     x4xad  = XAD,但時長標籤來自 1 h MFA
#   並與 base / c2(+AD)/ c6(XAD)/ 48h 並列。主實驗的 durations 與清單完全不動(X4 用獨立的 symlink 資料樹)。
#   nohup bash x4_aligner_1h.sh > ~/hakka_tts/runs/x4.log 2>&1 &
#   REPORT_ONLY=1 bash x4_aligner_1h.sh        # 不訓練,只用已完成的模型出報告
#   JOBS=8 ...                                 # MFA 平行工作數(預設 4)
# 用 GPU:12 次訓練(2 系統 × 2 腔 × 3 種子),每次約 2 小時;每個模型評估約 30–40 分鐘。完成的步驟會自動跳過。
# 報告:~/hakka_tts/exp_c/x4/<腔>/label_quality.md(標籤品質)、~/hakka_tts/exp_c/x4/x4_<腔>_vs_base.md
set -uo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
H=${HAKKA_ROOT:-$HOME/hakka_tts}; M=${MATCHA_ROOT:-$HOME/Matcha-TTS}; E=$H/exp_c/eval; C3=$H/exp_c/c3
G=$H/exp_c/g1; O=$H/exp_c/x4
DP_OVR="model.encoder.duration_predictor_params.kernel_size=5 model.encoder.duration_predictor_params.filter_channels_dp=512"
SEEDS="1234 123 2026"
JOBS=${JOBS:-4}
TRAIN_SPLIT=${TRAIN_SPLIT:-train_full}
mkdir -p "$O" "$H/runs"
source ~/miniconda3/etc/profile.d/conda.sh
act() { set +u; conda activate "$1"; set -u; }
act tts
has_step40k() { compgen -G "$M/logs/train/$1/runs/*/checkpoints/step_0039999.ckpt" >/dev/null || \
                compgen -G "$M/logs/train/$1/runs/*/checkpoints/step_0040000.ckpt" >/dev/null; }
sfx() { [ "$1" = 1234 ] && echo "" || echo "_s$1"; }
lex_of() { [ "$1" = hailu ] && echo "$H/dict/hailu_lexicon.txt" || echo "$H/dict/hakka_lexicon.txt"; }
wait_mfa() { while pgrep -f "mfa (align|train) " > /dev/null; do sleep 60; done; }

exp_of() {  # 系統 腔 種子 → 實驗(run_name)
  case $1 in
    base)  echo fair1h_joint_$2_scratch_s$3 ;;
    c2)    echo c2_mfadur_fair1h_joint_$2_scratch_s$3 ;;
    c6)    echo c6_tr_dpbig_fair1h_joint_$2_s$3 ;;
    x4ad)  echo x4_ad_fair1h_joint_$2_s$3 ;;
    x4xad) echo x4_xad_fair1h_joint_$2_s$3 ;;
  esac
}
extra_of() { case $1 in c6|x4xad) echo "--data-yaml $M/configs/data/fair1h_joint_$2.yaml" ;; esac; }   # 遷移模型用目標腔 mel 統計量
val_of() {  # 系統 腔 → 驗證集清單(選取整用;X4 系統用 1 h MFA 的時長)
  case $1 in
    x4ad|x4xad) echo "$O/$2/lists/val_mfadur.txt" ;;
    *) local v; v=$(python -c "import yaml;print(yaml.safe_load(open('$M/configs/data/fair1h_joint_$2_mfadur.yaml'))['valid_filelist_path'])")
       [[ $v = /* ]] && echo "$v" || echo "$M/$v" ;;
  esac
}

# ---------------- 1. 1 h MFA 與時長標籤
if [ "${REPORT_ONLY:-0}" != 1 ]; then
  for d in sixian hailu; do
    X=$O/$d; LEX=$(lex_of "$d"); MODEL=$X/acoustic_1h_$d.zip
    act tts
    echo "===== $d:建立 X4 資料樹($(date '+%m-%d %H:%M'))"
    python "$HERE/x4_prepare.py" --dialect "$d" --out "$X" || { echo "★ $d 資料樹建立失敗"; continue; }
    act mfa
    wait_mfa                                   # 前一次中斷時留下的 mfa 若還在跑,等它寫完模型
    if [ ! -f "$MODEL" ]; then
      echo "===== $d:mfa train(只用 1 h 訓練子集)($(date '+%m-%d %H:%M'))"
      mfa train "$X/x4train_$d" "$LEX" "$MODEL" --clean -j "$JOBS" > "$X/mfa_train.log" 2>&1 \
          || { echo "★ mfa train 失敗,看 $X/mfa_train.log 最後幾行:"; tail -5 "$X/mfa_train.log"; continue; }
    fi
    for split in train val; do
      if ! compgen -G "$X/tg/$split/*/*.TextGrid" > /dev/null; then
        wait_mfa
        echo "===== $d:用 1 h MFA 對齊 $split($(date '+%m-%d %H:%M'))"
        mfa align "$X/x4${split}_$d" "$LEX" "$MODEL" "$X/tg/$split" --clean --single_speaker -j "$JOBS" > "$X/mfa_align_$split.log" 2>&1 \
            || echo "★ $split 對齊失敗,看 $X/mfa_align_$split.log"
      fi
    done
    act tts
    if ! grep -q offset_tg_dir "$X/lists/mfadur_report.json" 2>/dev/null; then   # 舊版(裁切點用 1 h MFA 推算)的清單要重做
      # 時長轉換的裁切對位設定沿用主實驗(讀主實驗留下的 mfadur_report.json);
      # 裁切點一律用主實驗的 TextGrid 推算(22 kHz 音檔當初就是依它裁切),時長才用 1 h MFA 的邊界
      OFF=$(python - "$M/data/hakka/joint/$d/mfadur_report.json" <<'PY'
import json, sys
a = json.load(open(sys.argv[1], encoding="utf-8"))["args"]
out = ["--offset-mode", a["offset_mode"]]
for k in ("offset_csv", "margin_ms", "orig_root", "max_lead_ms", "min_lead_ms", "tail_tol_ms"):
    if a.get(k) is not None:
        out += ["--" + k.replace("_", "-"), str(a[k])]
print(" ".join(out))
PY
) || { echo "★ 讀不到主實驗的 mfadur_report.json,無法沿用裁切設定"; continue; }
      echo "  沿用主實驗的時長轉換設定:$OFF"
      MAIN_TG=$(python -c "import sys; sys.path.insert(0, '$HERE'); from hakka_c import config as C; print(C.textgrid_dir('$d'))")
      rm -f "$X/lists/"*_mfadur.txt
      python "$HERE/c2_mfa2durations.py" --dialect "$d" --tg-dir "$X/tg" --list-dir "$X/lists" --out-dir "$X/wav22k/durations" $OFF \
          --offset-tg-dir "$MAIN_TG" | grep -E "句可用|留白|排除原因"
    fi
    # 可用句數要接近主實驗(主實驗的清單本來就已經過同樣的過濾),太少代表轉換有問題,不要拿去訓練
    python - "$X/lists/mfadur_report.json" <<'PY' || { echo "★ $d:X4 時長可用句數太少,停在這裡(看上面的排除原因)"; rm -f "$X/lists/"*_mfadur.txt; continue; }
import json, sys
r = json.load(open(sys.argv[1], encoding="utf-8"))["lists"]
bad = {k: v for k, v in r.items() if v["kept"] < 0.9 * v["total"]}
print("  可用句數:" + "、".join(f"{k} {v['kept']}/{v['total']}" for k, v in r.items()))
sys.exit(1 if bad else 0)
PY
    python "$HERE/x4_compare.py" --dialect "$d" --new-tg "$X/tg/train" | tee "$X/label_quality.md"
  done
fi

# ---------------- 2. 訓練(與主實驗相同的設定,只換訓練 / 驗證清單)
train() {
  local k=$1 d=$2 s=$3 exp cfg ovr="" X=$O/$2
  exp=$(exp_of "$k" "$d" "$s")
  has_step40k "$exp" && return 0
  [ -f "$X/lists/train_mfadur.txt" ] || { echo "★ 沒有 X4 時長清單:$X/lists/train_mfadur.txt"; return 1; }
  case $k in
    x4ad)  cfg=c2_mfadur_fair1h_joint_${d}_scratch_s1234 ;;
    x4xad) cfg=c2_mfadur_fair1h_joint_${d}_transfer_s1234; ovr=$DP_OVR ;;
  esac
  local data="data.train_filelist_path=$X/lists/train_mfadur.txt data.valid_filelist_path=$X/lists/val_mfadur.txt"
  echo "----- $exp 設定檢查(seed: $s、清單指向 X4$([ "$k" = x4xad ] && echo '、kernel_size: 5、filter_channels_dp: 512、transfer_from'))"
  (cd "$M" && python matcha/train.py experiment="$cfg" seed="$s" run_name="$exp" $ovr $data --cfg job 2>&1 \
      | grep -n -E "^seed|train_filelist_path|valid_filelist_path|load_durations|kernel_size|filter_channels_dp|transfer_from|max_steps" | head -10)
  echo "===== 訓練 $exp($(date '+%m-%d %H:%M'))"
  (cd "$M" && python matcha/train.py experiment="$cfg" seed="$s" run_name="$exp" $ovr $data) > "$H/runs/$exp.log" 2>&1
  grep -E "^\[transfer\]" "$H/runs/$exp.log" | head -1
  has_step40k "$exp" || echo "★ 訓練失敗:$exp(看 $H/runs/$exp.log)"
}
if [ "${REPORT_ONLY:-0}" != 1 ]; then
  act tts
  for k in x4ad x4xad; do
    for d in sixian hailu; do
      for s in $SEEDS; do train "$k" "$d" "$s"; done
    done
  done
fi

# ---------------- 3. 評估、驗證集選取整(同 c7_seeds.sh / x2_ablation.sh)
choose() {
  local k=$1 d=$2 s=$3 exp val name pick dir
  exp=$(exp_of "$k" "$d" "$s")
  has_step40k "$exp" || { echo "★ 沒有第 40000 步:$exp" >&2; return 1; }
  dir=${k}_${d}$(sfx "$s")_step40000
  if [ ! -f "$E/$dir/syllables_wave.csv" ]; then
    bash "$HERE/d1_pipeline.sh" "$d" "$dir" "$exp" --which step40000 $(extra_of "$k" "$d") >&2 || { echo "★ 評估失敗:$dir" >&2; return 1; }
    act tts
  fi
  val=$(val_of "$k" "$d")
  name=c6val_${k}_${d}$(sfx "$s")
  [ -f "$C3/$name/recs.pkl" ] || python "$HERE/c3_target_check.py" --exp "$exp" --dialect "$d" --which step40000 \
      --filelist "$val" --name "$name" > "$H/runs/c6_${name}.log" 2>&1 || { echo "★ 驗證集診斷失敗:$name" >&2; return 1; }
  pick=$(python "$HERE/c6_pick_rounding.py" --val "$C3/$name/recs.pkl" | tail -1)
  if [ "$pick" = cum ]; then
    dir=${dir}_cum
    if [ ! -f "$E/$dir/syllables_wave.csv" ]; then
      bash "$HERE/d1_pipeline.sh" "$d" "$dir" "$exp" --which step40000 --rounding cum $(extra_of "$k" "$d") >&2 \
          || { echo "★ 評估失敗:$dir" >&2; return 1; }
      act tts
    fi
  fi
  echo "  [$k $d s$s] 選 $pick → $dir" >&2
  echo "$dir"
}

for d in sixian hailu; do
  act tts
  echo "===== $d:評估與驗證集選取整($(date '+%m-%d %H:%M'))"
  [ -f "$O/$d/label_quality.md" ] && sed -n '/^|/p;/邊界/p' "$O/$d/label_quality.md"
  declare -A DIRS=()
  for k in base c2 c6 x4ad x4xad; do
    out=""
    for s in $SEEDS; do x=$(choose "$k" "$d" "$s") && out+="$E/$x,"; done
    DIRS[$k]=${out%,}
    echo "  $k:$(tr ',' '\n' <<< "${DIRS[$k]}" | grep -c .) 個種子"
  done
  sys_args=()
  for k in base c2 c6 x4ad x4xad; do [ -n "${DIRS[$k]}" ] && sys_args+=(--sys "$k=${DIRS[$k]}"); done
  [ -f "$E/big48h_$d/syllables_wave.csv" ] && sys_args+=(--sys "48h=$E/big48h_$d")
  echo "===== $d:時長命中率、ρ、MCD(c2 / c6 = 主實驗;x4ad / x4xad = 1 h MFA 的標籤)"
  for b in base c2 c6; do
    [ -n "${DIRS[$b]}" ] || continue
    python "$HERE/x1_overlap.py" --dialect "$d" --base "$b" "${sys_args[@]}" --out "$O/x4_${d}_vs_${b}.md" \
        | sed -n "/## 全部/,/## 排除/p" | grep -v "## 排除"
  done
  echo "===== $d:G1 聲調辨識(與 base 配對,三種子合併)"
  if [ ! -f "$G/real_${d}_${TRAIN_SPLIT}.csv" ] || [ ! -f "$G/real_${d}_test.csv" ]; then
    echo "★ 找不到真人特徵(先跑 g1_run.sh),略過 G1"
  else
    IFS=, read -r -a A <<< "${DIRS[base]}"
    for k in x4ad x4xad; do
      IFS=, read -r -a B <<< "${DIRS[$k]}"
      [ ${#A[@]} -eq 3 ] && [ ${#B[@]} -eq 3 ] || { echo "  ($k:種子不齊,略過)"; continue; }
      for x in "${A[@]}" "${B[@]}"; do
        [ -f "$x/syl_f0.csv" ] || python "$HERE/g1_tone_feats.py" synth --dialect "$d" --eval-dir "$x" > /dev/null 2>&1 \
            || echo "★ F0 特徵失敗:$x"
      done
      sa=(); ca=(); i=0
      for s in $SEEDS; do
        sa+=(--sys "base_s$s=${A[$i]}/syl_f0.csv" --sys "${k}_s$s=${B[$i]}/syl_f0.csv"); ca+=(--compare "base_s$s" "${k}_s$s"); i=$((i + 1))
      done
      gout=$O/g1_${d}_${k}_vs_base
      python "$HERE/g1_tone_eval.py" --dialect "$d" --train "$G/real_${d}_${TRAIN_SPLIT}.csv" --ceiling "$G/real_${d}_test.csv" \
          "${sa[@]}" "${ca[@]}" --out-dir "$gout" > /dev/null 2>&1 || { echo "★ G1 失敗:$k"; continue; }
      echo "  $k:與 base 的聲調辨識率差(合併):$(grep '合併' "$gout/g1_summary.md" | cut -d'|' -f3 | xargs)"
    done
  fi
  unset DIRS
done
echo "完成($(date '+%m-%d %H:%M'))"
