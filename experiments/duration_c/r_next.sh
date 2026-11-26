#!/usr/bin/env bash
# 第三種架構的後續三件事,一行跑完(約 18–20 小時,中途不用管):
#   1. 語音辨識檢查(r_asr.sh):Raon-OpenTTS 合成的客語辨識得出來嗎?和 Matcha-TTS、VITS 並列     約 0.5–1 小時(GPU)
#   2. 相同頻寬的 MCD(r_mcd16.py):全部系統都濾掉 8 kHz 以上再算,音質才能和 16 kHz 的 Raon 比   與第 3 步同時跑(CPU)
#   3. Raon-OpenTTS 訓練延長到 4 萬步(r_run.sh TAG=_40k):2 萬步時驗證集曲線還在上升          約 16–18 小時(GPU)
#   最後把 2 萬步、4 萬步的 Raon 和其他系統並列,MCD 用相同頻寬的版本 → ~/hakka_tts/exp_c/r/final_<腔調>.md
#
#   nohup bash r_next.sh > ~/hakka_tts/runs/r_next.log 2>&1 &
set -uo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
H=${HAKKA_ROOT:-$HOME/hakka_tts}; E=$H/exp_c/eval; O=$H/exp_c/r
JOBS=${JOBS:-4}
source ~/miniconda3/etc/profile.d/conda.sh
act() { set +u; conda activate "$1"; set -u; }
source "$HERE/sys_dirs.sh"

echo "===== 1. 語音辨識檢查($(date '+%m-%d %H:%M'))"
SYSTEMS="base c6 vbase vxad raon" bash "$HERE/r_asr.sh" || echo "★ 語音辨識檢查失敗(繼續做後面的步驟)"

echo "===== 2. 相同頻寬的 MCD(背景執行,log 在 $O/mcd16_<腔調>.log)"
act tts
for d in sixian hailu; do
  dirs=()
  for k in base c6 vbase vxad raon; do x=$(sys_dirs "$k" "$d"); [ -n "$x" ] && dirs+=(${x//,/ }); done
  python "$HERE/r_mcd16.py" --dialect "$d" --dirs "${dirs[@]}" --jobs "$JOBS" > "$O/mcd16_$d.log" 2>&1 &
done

echo "===== 3. Raon-OpenTTS 延長到 4 萬步($(date '+%m-%d %H:%M'))"
TAG=_40k STEPS=40000 SAVE="5000,10000,20000,30000,40000" bash "$HERE/r_run.sh"
wait
act tts
for d in sixian hailu; do
  grep -E "檢查|★" "$O/mcd16_$d.log"
  x=$(sys_dirs raon_40k "$d")
  [ -n "$x" ] && python "$HERE/r_mcd16.py" --dialect "$d" --dirs ${x//,/ } --jobs "$JOBS" >> "$O/mcd16_$d.log" 2>&1
done
SYSTEMS="base c6 vbase vxad raon raon_40k" bash "$HERE/r_asr.sh" || echo "★ 4 萬步的語音辨識檢查失敗"

echo "===== 總表($(date '+%m-%d %H:%M'))"
act tts
for d in sixian hailu; do
  args=()
  for k in base c6 vbase vxad raon raon_oracle raon_40k raon_40k_oracle; do
    x=$(sys_dirs "$k" "$d"); [ -n "$x" ] && args+=(--sys "$k=$x")
  done
  echo "##### $d(MCD 為 8 kHz 以下)"
  python "$HERE/x1_overlap.py" --dialect "$d" --base base "${args[@]}" --mcd-csv mcd16.csv --out "$O/final_$d.md" \
      | sed -n "/## 全部/,/## 排除/p" | grep -v "## 排除"
done
echo "完成($(date '+%m-%d %H:%M'))"
