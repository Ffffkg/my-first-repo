#!/usr/bin/env bash
# A1:用已經辨識好的結果重算報告(不用 GPU,幾秒鐘):兩個腔調各一份,再加合併兩腔的一份。只印出「共同句數」之後的部分。
#   bash a1_report.sh                       # 預設模型與語言;換過模型就帶同樣的環境變數:MODEL=... OMNI_LANG=... BACKEND=...
set -uo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
H=${HAKKA_ROOT:-$HOME/hakka_tts}; O=$H/exp_c/a1
EXTRA=(--backend "${BACKEND:-omni}")
[ -n "${MODEL:-}" ] && EXTRA+=(--model "$MODEL")
[ -n "${OMNI_LANG:-}" ] && EXTRA+=(--lang "$OMNI_LANG")
for d in ${DIALECTS:-sixian hailu}; do
  echo "##### $d"
  python "$HERE/a1_asr_check.py" analyze --dialect "$d" --out-dir "$O/$d" "${EXTRA[@]}" | sed -n '/^共同句數/,$p'
done
echo "##### 合併"
python "$HERE/a1_asr_check.py" pool --out-root "$O" --dialects ${DIALECTS:-sixian hailu} "${EXTRA[@]}" | sed -n '/^共同句數/,$p'
