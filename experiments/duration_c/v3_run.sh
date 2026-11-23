#!/usr/bin/env bash
# V3:為什麼 VITS 比 Matcha-TTS 差?為什麼 VITS +AD 在海陸變差?
#   V3a(不需要 GPU):兩架構並列,拆成「時長預測」與「發音忠實度」
#   V3b(GPU 幾分鐘):Base 模型自己找的對齊(MAS)和 MFA 時長差多少
#   V2 (不需要 GPU):VITS 各系統的訓練損失與訓練句數
#   bash v3_run.sh
set -uo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
H=${HAKKA_ROOT:-$HOME/hakka_tts}; O=$H/exp_c/v3
mkdir -p "$O"
source ~/miniconda3/etc/profile.d/conda.sh
set +u; conda activate tts; set -u
echo "===== V3a:時長預測 vs 發音忠實度"
python "$HERE/v3_arch_compare.py" | tee "$O/v3a.md"
echo; echo "===== V2:VITS 訓練損失與訓練句數"
python "$HERE/v2_vits_diag.py" | tee "$O/v2.md"
echo; echo "===== V3b:Base 模型的 MAS vs MFA($(date '+%m-%d %H:%M'))"
python "$HERE/v3_mas_vs_mfa.py" 2> "$O/v3b.log" | tee "$O/v3b.md" || { echo "★ V3b 失敗,看 $O/v3b.log"; tail -5 "$O/v3b.log"; }
echo "完成($(date '+%m-%d %H:%M'))"
