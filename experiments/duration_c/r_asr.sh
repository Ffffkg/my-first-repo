#!/usr/bin/env bash
# R5:語音辨識檢查(A1 的流程)加入 VITS 與 Raon-OpenTTS:合成的客語辨識得出來嗎?入聲字有沒有特別吃虧?
#   用 asr 環境的 Omnilingual ASR(和論文 Table IV 相同);辨識結果存在 ~/hakka_tts/exp_c/r/a1/<腔調>/,
#   第一次會複製論文 A1 已辨識好的結果(base、c6…),只辨識新的系統;報告是 a1_*_pooled.md(兩腔合併)。
#   bash r_asr.sh                       # 預設系統:base c6 vbase vxad raon
#   SYSTEMS="base c6 raon raon_40k" bash r_asr.sh
set -uo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
H=${HAKKA_ROOT:-$HOME/hakka_tts}; E=$H/exp_c/eval; O=$H/exp_c/r/a1
SYSTEMS=${SYSTEMS:-"base c6 vbase vxad raon"}
source ~/miniconda3/etc/profile.d/conda.sh
act() { set +u; conda activate "$1"; set -u; }
source "$HERE/sys_dirs.sh"
act asr
python -c "import omnilingual_asr" 2>/dev/null || { echo "★ asr 環境 import omnilingual_asr 失敗(先跑 a1_setup_env.sh)"; exit 1; }
for d in sixian hailu; do
  mkdir -p "$O/$d"
  for f in "$H"/exp_c/a1/"$d"/transcripts_*.csv; do      # 沿用論文 A1 已辨識的結果(同一模型、同一設定)
    [ -f "$f" ] && [ ! -f "$O/$d/$(basename "$f")" ] && cp "$f" "$O/$d/"
  done
  act tts
  args=()
  for k in $SYSTEMS; do x=$(sys_dirs "$k" "$d"); [ -n "$x" ] && args+=(--sys "$k=$x"); done
  act asr
  echo "##### $d:$(( ${#args[@]} / 2 )) 個系統"
  python "$HERE/a1_asr_check.py" run --dialect "$d" --out-dir "$O/$d" --batch "${BATCH:-8}" "${args[@]}" \
      | grep -E "句;系統|待辨識|→" || { echo "★ $d 辨識失敗"; exit 1; }
done
python "$HERE/a1_asr_check.py" pool --out-root "$O" | sed -n '/## 摘要/,/^## [^摘]/p'
echo "完整報告:$(ls "$O"/a1_*_pooled.md)"
