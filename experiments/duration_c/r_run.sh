#!/usr/bin/env bash
# R:第三種架構 Raon-OpenTTS(擴散 Transformer + 流匹配,沒有時長預測器,只給整句總長)。
#   和 Matcha-TTS / VITS 的 Base 用同一份 1 小時訓練清單、同樣三個種子、同一批測試句與 D1 量法。
#   差別(架構本身決定,無法拉平):從官方英文預訓練權重微調(1 小時資料從頭訓練學不起來);
#   語者用「同一位語者的一句訓練句」當聲音提示;整句總長由語速估(主要結果)或給真人長度(診斷)。
#
#   先裝好:bash r_setup.sh
#   nohup bash r_run.sh > ~/hakka_tts/runs/r.log 2>&1 &
#   DIALECTS="sixian hailu"(預設)  STEPS=20000  SAVE="2500,5000,10000,15000,20000"(在驗證集上從中選一個)
#   REPORT_ONLY=1 bash r_run.sh    # 不訓練,只出報告
# 用 GPU:煙霧測試會印出每步秒數;預估每次訓練 2–4 小時 × 6 次,加上驗證集選步數與測試評估,約 1.5–2 天。
set -uo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
H=${HAKKA_ROOT:-$HOME/hakka_tts}; RAON=${RAON_ROOT:-$H/raon}
E=$H/exp_c/eval; O=$H/exp_c/r
SEEDS="1234 123 2026"; DIALECTS=${DIALECTS:-"sixian hailu"}; STEPS=${STEPS:-20000}
SAVE=${SAVE:-"2500,5000,10000,15000,20000"}; VAL_PER_SPK=${VAL_PER_SPK:-20}
mkdir -p "$O" "$H/runs"
source ~/miniconda3/etc/profile.d/conda.sh
act() { set +u; conda activate "$1"; set -u; }
sfx() { [ "$1" = 1234 ] && echo "" || echo "_s$1"; }
conda env list | awk '{print $1}' | grep -qx raon || { echo "★ 先執行:bash $HERE/r_setup.sh"; exit 1; }
act raon

# ---------------- 0. 資料(兩腔一起做,詞表共用)
need_prep=0
for d in sixian hailu; do for sp in train val test; do [ -s "$RAON/data/$d/$sp.jsonl" ] || need_prep=1; done; done
[ -f "$RAON/vocab_hakka.txt" ] || need_prep=1
if [ $need_prep = 1 ]; then
  echo "===== 準備資料($(date '+%m-%d %H:%M'))"
  python "$HERE/r_prep.py" || { echo "★ 資料準備失敗"; exit 1; }
fi

# ---------------- 1. 煙霧測試:訓練 30 步 + 合成 1 句
if [ "${REPORT_ONLY:-0}" != 1 ] && [ ! -f "$O/smoke_ok" ]; then
  d=${DIALECTS%% *}
  echo "===== 煙霧測試($(date '+%m-%d %H:%M'))"
  rm -rf "$RAON/runs/raon_smoke" "$E/raon_smoke"
  python "$HERE/r_train.py" --exp raon_smoke --dialect "$d" --max-steps 30 --save-steps 30 --log-every 10 > "$O/smoke_train.log" 2>&1
  grep -E "訓練句數|step 30|★" "$O/smoke_train.log"
  [ -f "$RAON/runs/raon_smoke/model_30.pt" ] || { echo "★ 煙霧測試訓練失敗,看 $O/smoke_train.log"; tail -5 "$O/smoke_train.log"; exit 1; }
  python "$HERE/r_synth.py" --exp raon_smoke --which step30 --dialect "$d" --name raon_smoke --per-spk 1 \
      --gen-seeds 1234 --no-mcd > "$O/smoke_synth.log" 2>&1
  n=$(ls "$E/raon_smoke/wav" 2>/dev/null | wc -l)
  [ "$n" -ge 1 ] || { echo "★ 煙霧測試合成失敗,看 $O/smoke_synth.log"; tail -5 "$O/smoke_synth.log"; exit 1; }
  echo "  合成 $n 個音檔 ✓"
  rm -rf "$RAON/runs/raon_smoke" "$E/raon_smoke"
  touch "$O/smoke_ok"
fi

# ---------------- 2. 訓練
if [ "${REPORT_ONLY:-0}" != 1 ]; then
  for d in $DIALECTS; do for s in $SEEDS; do
    exp=raon_fair1h_${d}_s$s
    [ -f "$RAON/runs/$exp/model_$STEPS.pt" ] && continue
    echo "===== 訓練 $exp($(date '+%m-%d %H:%M'))"
    python "$HERE/r_train.py" --exp "$exp" --dialect "$d" --seed "$s" --max-steps "$STEPS" --save-steps "$SAVE" \
        > "$H/runs/$exp.log" 2>&1
    grep -E "訓練句數|★" "$H/runs/$exp.log" | head -2
    grep -E "^step" "$H/runs/$exp.log" | tail -1
    [ -f "$RAON/runs/$exp/model_$STEPS.pt" ] || echo "★ 訓練失敗:$exp(看 $H/runs/$exp.log)"
  done; done
fi

# ---------------- 3. 驗證集選步數(每個種子各自選;全部母音 ±30 ms 命中率)
pick_step() {
  local d=$1 s=$2 exp=raon_fair1h_$1_s$2 args=() st name
  for st in ${SAVE//,/ }; do
    [ -f "$RAON/runs/$exp/model_$st.pt" ] || continue
    name=rval_${d}$(sfx "$s")_step$st
    if [ ! -f "$E/$name/syllables_wave.csv" ]; then
      bash "$HERE/r_pipeline.sh" "$d" "$name" "$exp" --which "step$st" --split val --per-spk "$VAL_PER_SPK" \
          --gen-seeds 1234 --no-mcd > "$H/runs/$name.log" 2>&1 || echo "★ 驗證失敗:$name(看 $H/runs/$name.log)" >&2
      act raon
    fi
    args+=("$st=$E/$name")
  done
  [ ${#args[@]} -gt 0 ] || return 1
  python "$HERE/r_pick.py" --dirs "${args[@]}" > "$O/pick_${d}$(sfx "$s").md" || return 1
  tail -1 "$O/pick_${d}$(sfx "$s").md"
}

# ---------------- 4. 測試集評估 + 報告
find_dir() {  # 系統 腔 種子 步數 → 既有的評估資料夾
  local sf n; sf=$(sfx "$3")
  for n in "${1}_${2}${sf}_step${4}_cum" "${1}_${2}${sf}_step${4}"; do
    [ -f "$E/$n/syllables_wave.csv" ] && { echo "$E/$n"; return 0; }
  done
  return 1
}
for d in $DIALECTS; do
  echo "===== $d:驗證集選步數、測試集評估($(date '+%m-%d %H:%M'))"
  R1=""; R2=""
  for s in $SEEDS; do
    exp=raon_fair1h_${d}_s$s
    st=$(pick_step "$d" "$s") || { echo "★ $exp 沒有可用的 checkpoint"; continue; }
    echo "  [$exp] 驗證集選第 $st 步"; sed -n '1,8p' "$O/pick_${d}$(sfx "$s").md" | grep -E "^\| [0-9]" | sed 's/^/    /'
    for tot in rate oracle; do
      name=raon_${d}$(sfx "$s")_step$st$([ $tot = oracle ] && echo _oracle)
      if [ ! -f "$E/$name/syllables_wave.csv" ]; then
        bash "$HERE/r_pipeline.sh" "$d" "$name" "$exp" --which "step$st" --total $tot > "$H/runs/$name.log" 2>&1 \
            || { echo "★ 評估失敗:$name(看 $H/runs/$name.log)"; act raon; continue; }
        act raon
      fi
      [ $tot = rate ] && R1+="$E/$name," || R2+="$E/$name,"
    done
  done
  act tts
  sys_args=()
  for spec in "base:base:40000" "xad:c6:40000" "vits:vbase:200000" "vits_xad:vxad:200000"; do
    IFS=: read -r lab k st <<< "$spec"
    out=""; for s in $SEEDS; do x=$(find_dir "$k" "$d" "$s" "$st") && out+="$x,"; done
    [ -n "$out" ] && sys_args+=(--sys "$lab=${out%,}") || echo "  (找不到 $k 的評估結果,報告略過)"
  done
  [ -n "$R1" ] && sys_args+=(--sys "raon=${R1%,}")
  [ -n "$R2" ] && sys_args+=(--sys "raon_oracle=${R2%,}")
  echo "===== $d:入聲 / 舒聲命中率、ρ、MCD(與 Matcha-TTS Base 比較)"
  python "$HERE/x1_overlap.py" --dialect "$d" --base base "${sys_args[@]}" --out "$O/r_${d}.md" \
      | sed -n "/## 全部/,/## 排除/p" | grep -v "## 排除"
  for x in ${R1//,/ } ${R2//,/ }; do
    python -c "import json,sys; j=json.load(open(sys.argv[1]+'/raon_info.json')); print(f\"  {sys.argv[1].split('/')[-1]}:合成長度/真人長度 {j['len_ratio_mean']:.3f}(平均絕對差 {j['len_ratio_abs_err_pct']:.1f}%)\")" "$x"
  done
done
echo "完成($(date '+%m-%d %H:%M'))"
