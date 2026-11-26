#!/usr/bin/env bash
# Raon-OpenTTS 版的 d1_pipeline.sh:合成(r_synth.py)→ MFA corpus → mfa align → 波形比較(d1_compare.py)
#   bash r_pipeline.sh <腔調> <輸出名稱> <實驗名稱> [r_synth.py 的其他參數...]
#   bash r_pipeline.sh sixian raon_sixian_step20000 raon_fair1h_sixian_s1234 --which step20000
set -euo pipefail
dialect=$1; name=$2; exp=$3; shift 3
HERE=$(cd "$(dirname "$0")" && pwd)
HAKKA=${HAKKA_ROOT:-$HOME/hakka_tts}
E=$HAKKA/exp_c/eval
LEX=hakka_lexicon.txt; [ "$dialect" = hailu ] && LEX=hailu_lexicon.txt
source ~/miniconda3/etc/profile.d/conda.sh

conda activate raon
python "$HERE/r_synth.py" --exp "$exp" --dialect "$dialect" --name "$name" "$@"
python "$HERE/d1_prepare_mfa.py" --eval-dir "$E/$name" --dialect "$dialect" | head -1

conda activate mfa
echo "mfa align …(log:$E/$name/mfa_align.log)"
mfa align "$E/$name/mfa_corpus" "$HAKKA/dict/$LEX" "$HAKKA/aligned/$dialect/acoustic" "$E/$name/mfa_tg" \
    --clean --single_speaker > "$E/$name/mfa_align.log" 2>&1

conda activate tts
python "$HERE/d1_compare.py" --eval-dir "$E/$name" --dialect "$dialect" | sed -n '/## 3/,$p'
