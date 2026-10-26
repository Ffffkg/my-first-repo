#!/usr/bin/env bash
# A1 天花板:真人錄音過同一個聲碼器重建(copy-synthesis)→ 用同一個 ASR 辨識 → 放進 A1 報告當 copy 那一列。
#   copy 的數字 = 「聲學模型與時長都完美」時各項指標能到多少,合成系統要跟它比才知道還有多少進步空間。
#   先跑過 a1_run.sh 之後:  bash a1_copy.sh        (用 GPU,每個腔調約 1–3 分鐘;辨識結果會存起來,重跑不會重複辨識)
set -uo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
H=${HAKKA_ROOT:-$HOME/hakka_tts}; E=$H/exp_c/eval; O=$H/exp_c/a1
DIALECTS=${DIALECTS:-"sixian hailu"}
EXTRA=(--backend "${BACKEND:-omni}")
[ -n "${MODEL:-}" ] && EXTRA+=(--model "$MODEL")
[ -n "${OMNI_LANG:-}" ] && EXTRA+=(--lang "$OMNI_LANG")
source ~/miniconda3/etc/profile.d/conda.sh
act() { set +u; conda activate "$1"; set -u; }
for d in $DIALECTS; do
  echo "##### $d"
  act tts
  python "$HERE/a1_copysynth.py" --dialect "$d" --like "$E/base_${d}_step40000" --out "$O/copysynth_$d" || exit 1
  act asr
  python "$HERE/a1_asr_check.py" run --dialect "$d" --out-dir "$O/$d" --batch "${BATCH:-8}" --sys "copy=$O/copysynth_$d" "${EXTRA[@]}" > /dev/null || exit 1
done
act tts
bash "$HERE/a1_report.sh"
