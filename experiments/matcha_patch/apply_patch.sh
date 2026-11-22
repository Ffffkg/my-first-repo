#!/usr/bin/env bash
# 把「音節→音素雙層時長預測器」套用到 ~/Matcha-TTS。
#   1. 先確認要改的兩個檔案與 experiments/matcha_snapshot 完全相同(避免蓋掉你之後的修改)
#   2. 原檔備份成 *.presyl,再複製修改版與兩個新檔
#   還原:bash apply_patch.sh --revert
# 修改內容:text_encoder.py 多兩個參數(syl_dur、syl_nasal_rule),syl_dur 為 None 時行為與原本完全相同;
#          matcha_tts.py 在時長損失後加上音節總長與音節內比例的損失(syl_dur 為 None 時不加)。
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
SNAP=$HERE/../matcha_snapshot
M=${MATCHA_ROOT:-$HOME/Matcha-TTS}
FILES="matcha/models/components/text_encoder.py matcha/models/matcha_tts.py"
NEW="matcha/models/components/syllable_duration.py matcha/models/components/hakka_syllables.py"
if [ "${1:-}" = "--revert" ]; then
  for f in $FILES; do [ -f "$M/$f.presyl" ] && cp "$M/$f.presyl" "$M/$f" && echo "還原 $f"; done
  for f in $NEW; do rm -f "$M/$f"; done
  exit 0
fi
for f in $FILES; do
  if cmp -s "$M/$f" "$HERE/$f"; then echo "已經是修改版:$f"; continue; fi
  if [ -f "$M/$f.presyl" ] && cmp -s "$M/$f.presyl" "$SNAP/$f"; then
    cp "$HERE/$f" "$M/$f"; echo "更新為最新修改版:$f(原檔備份仍是 $f.presyl)"; continue
  fi
  cmp -s "$M/$f" "$SNAP/$f" || { echo "★ $M/$f 與 snapshot 不同(之後又改過?),為了安全先停下來"; exit 1; }
  cp "$M/$f" "$M/$f.presyl"
  cp "$HERE/$f" "$M/$f"
  echo "套用 $f(原檔備份為 $f.presyl)"
done
for f in $NEW; do cp "$HERE/$f" "$M/$f"; echo "新增 $f"; done
echo "完成。檢查:python $HERE/check_syllable_duration.py"
