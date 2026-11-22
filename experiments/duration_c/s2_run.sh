#!/usr/bin/env bash
# S2:真人錄音上兩套 MFA 的邊界一致性(s2_boundary_check.py),並用已有的 S1 合成結果重算 S1 報告
#     (新增 (b) − (a) 的配對信賴區間、聲母／韻尾占比)。不需要 GPU,幾分鐘內完成。
#   bash s2_run.sh
set -uo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
H=${HAKKA_ROOT:-$HOME/hakka_tts}; E=$H/exp_c/eval; O=$H/exp_c/s1
source ~/miniconda3/etc/profile.d/conda.sh
set +u; conda activate tts; set -u
for d in sixian hailu; do
  echo "===== $d:兩套 MFA 的邊界一致性"
  python "$HERE/s2_boundary_check.py" --dialect "$d" --out "$O/s2_boundary_$d.md" || echo "★ 邊界檢查失敗:$d"
  dirs=$(ls -d "$E"/s1val_x4xad_${d}*_step40000* 2>/dev/null | tr '\n' ',')
  [ -n "$dirs" ] || { echo "★ 找不到 S1 的合成結果,先跑 s1_run.sh"; continue; }
  echo "===== $d:S1 報告(重算)"
  python "$HERE/s1_syllable_diag.py" --dialect "$d" --eval-dirs "${dirs%,}" --out "$O/s1_x4xad_$d.md" \
      | sed -n '/## 1/,/## 3/p' | grep -v "^## 3"
done
