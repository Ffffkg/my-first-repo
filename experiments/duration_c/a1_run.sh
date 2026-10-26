#!/usr/bin/env bash
# A1:用語音辨識模型(Omnilingual ASR,客語 hak_Latn)當「獨立於 MFA」的客觀檢查——
#   合成語音能不能被辨識得跟真人錄音一樣,入聲字有沒有特別吃虧(a1_asr_check.py 的三個指標)。
#   每個腔調辨識:真人測試錄音 + base / C2 / C6(1h,三種子,各自在驗證集選取整方式)+ 48h baseline 的合成語音。
#   先小試(只做四縣、10 句;主要是看 ASR 輸出的拼寫與格式):  DIALECTS=sixian NSENT=10 bash a1_run.sh
#   確認可用再完整跑:                                          bash a1_run.sh
#   換模型:MODEL=omniASR_LLM_3B_v2 bash a1_run.sh              (預設 omniASR_LLM_1B_v2:約 6 GiB 顯存、下載 8.5 GiB)
#   Whisper 客語微調版:BACKEND=whisper bash a1_run.sh          (預設模型 formospeech/whisper-large-v3-taiwanese-hakka)
# 用 GPU(要先跑過 a1_setup_env.sh)。辨識結果會存起來,中斷後直接重跑會接續;報告在 ~/hakka_tts/exp_c/a1/<腔調>/a1_*.md。
set -uo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
H=${HAKKA_ROOT:-$HOME/hakka_tts}; E=$H/exp_c/eval; C3=$H/exp_c/c3; O=$H/exp_c/a1
DIALECTS=${DIALECTS:-"sixian hailu"}
SEEDS="1234 123 2026"
BACKEND=${BACKEND:-omni}
EXTRA=()
[ -n "${MODEL:-}" ] && EXTRA+=(--model "$MODEL")
[ -n "${NSENT:-}" ] && EXTRA+=(--n-sent "$NSENT")
[ -n "${OMNI_LANG:-}" ] && EXTRA+=(--lang "$OMNI_LANG")       # 預設 hak_Latn;OMNI_LANG=none 表示不給語言碼
source ~/miniconda3/etc/profile.d/conda.sh
act() { set +u; conda activate "$1"; set -u; }
sfx() { [ "$1" = 1234 ] && echo "" || echo "_s$1"; }
dirs_of() {   # 系統 腔 → 逗號分隔的評估資料夾(各訓練種子;ceil / cum 由驗證集選,同 h4_tol.sh)
  local out="" s dir pick
  for s in $SEEDS; do
    dir=$1_$2$(sfx "$s")_step40000
    pick=$(python "$HERE/c6_pick_rounding.py" --val "$C3/c6val_$1_$2$(sfx "$s")/recs.pkl" 2>/dev/null | tail -1)
    [ "$pick" = cum ] && dir=${dir}_cum
    [ -d "$E/$dir/wav" ] && out+="$E/$dir,"
  done
  echo "${out%,}"
}
if [ "$BACKEND" = omni ]; then       # 先確認 asr 環境能 import(a1_setup_env.sh 沒跑完的話,在這裡就停,不要先花時間轉檔)
  act asr
  if ! python -c "import omnilingual_asr" 2> /tmp/a1_import.err; then
    echo "★ asr 環境裡 import omnilingual_asr 失敗,請先把 a1_setup_env.sh 跑完。錯誤的最後幾行:"; tail -4 /tmp/a1_import.err; exit 1
  fi
fi
for d in $DIALECTS; do
  act tts
  args=()
  for k in base c2 c6; do x=$(dirs_of "$k" "$d"); [ -n "$x" ] && args+=(--sys "$k=$x"); done
  [ -d "$E/big48h_$d/wav" ] && args+=(--sys "48h=$E/big48h_$d")
  if [ ${#args[@]} -eq 0 ]; then echo "★ $d:$E 底下找不到帶 wav/ 的評估資料夾"; continue; fi
  act asr
  echo "##### $d"
  python "$HERE/a1_asr_check.py" run --dialect "$d" --backend "$BACKEND" --out-dir "$O/$d" --batch "${BATCH:-8}" \
    ${EXTRA[@]+"${EXTRA[@]}"} "${args[@]}"
done
