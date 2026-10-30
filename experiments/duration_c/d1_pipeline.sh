#!/usr/bin/env bash
# 一行跑完:合成(c_eval --synth)→ MFA corpus → mfa align → 波形比較(d1_compare)
#
#   bash d1_pipeline.sh <腔調> <輸出名稱> <實驗名稱> [c_eval.py 的其他參數...]
#   bash d1_pipeline.sh sixian base_sixian_step40000 fair1h_joint_sixian_scratch_s1234 --which step40000
#
# 需要 conda 環境 tts(合成、分析)與 mfa(對齊);MFA 的完整輸出存在 <輸出資料夾>/mfa_align.log。
set -euo pipefail
dialect=$1; name=$2; exp=$3; shift 3
HERE=$(cd "$(dirname "$0")" && pwd)
HAKKA=${HAKKA_ROOT:-$HOME/hakka_tts}
E=$HAKKA/exp_c/eval
LEX=hakka_lexicon.txt; [ "$dialect" = hailu ] && LEX=hailu_lexicon.txt
source ~/miniconda3/etc/profile.d/conda.sh

conda activate tts
python "$HERE/c_eval.py" --exp "$exp" --dialect "$dialect" --synth --name "$name" "$@"
python "$HERE/d1_prepare_mfa.py" --eval-dir "$E/$name" --dialect "$dialect" | head -1

conda activate mfa
echo "mfa align …(log:$E/$name/mfa_align.log)"
mfa align "$E/$name/mfa_corpus" "$HAKKA/dict/$LEX" "$HAKKA/aligned/$dialect/acoustic" "$E/$name/mfa_tg" \
    --clean --single_speaker > "$E/$name/mfa_align.log" 2>&1

conda activate tts
python "$HERE/d1_compare.py" --eval-dir "$E/$name" --dialect "$dialect" | sed -n '/## 3/,$p'
