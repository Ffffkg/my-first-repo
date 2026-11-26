#!/usr/bin/env bash
# X3:「只用訓練集」重新訓練 MFA,重新量測母音核時長,確認論文主結果不是因為 MFA 看過測試錄音。只用 CPU。
#   論文主實驗的 MFA 用全部錄音(含驗證、測試句)訓練;這裡只用 benchmark/<腔>/train_full.tsv 的句子訓練,
#   再用它對齊 (1) 真人測試錄音 → 新的參考時長 (2) base / C2 / C6(三種子)與 48h 的合成語音 → 新的預測時長,
#   最後用同一張表(入聲 / 舒聲 ±30 ms 命中率、ρ、與 base 的差)和原本的量測並列。TTS 模型與合成語音都不重做。
#   nohup bash x3_mfa_trainonly.sh > ~/hakka_tts/runs/x3.log 2>&1 &
#   JOBS=8 nohup bash x3_mfa_trainonly.sh ...      # MFA 平行工作數(預設 4,留 CPU 給同時進行的 GPU 訓練)
# 時間:MFA 訓練每個腔調約數小時(約 2.2 萬句);對齊約 1 小時。完成的步驟會自動跳過,中斷後直接重跑。
# 會先等其他 mfa train / align 結束才開始自己的 MFA 步驟,避免與 x2_ablation.sh 的評估同時使用 MFA。
set -uo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
H=${HAKKA_ROOT:-$HOME/hakka_tts}; E=$H/exp_c/eval; C3=$H/exp_c/c3; O=$H/exp_c/x3
SEEDS="1234 123 2026"
JOBS=${JOBS:-4}
source ~/miniconda3/etc/profile.d/conda.sh
act() { set +u; conda activate "$1"; set -u; }
sfx() { [ "$1" = 1234 ] && echo "" || echo "_s$1"; }
lex_of() { [ "$1" = hailu ] && echo "$H/dict/hailu_lexicon.txt" || echo "$H/dict/hakka_lexicon.txt"; }
wait_mfa() { while pgrep -f "mfa (align|train) " > /dev/null; do sleep 60; done; }
dirs_of() {   # 同 h4_tol.sh:各訓練種子的評估資料夾,ceil / cum 由驗證集選
  local out="" s dir pick
  for s in $SEEDS; do
    dir=$1_$2$(sfx "$s")_step40000
    pick=$(python "$HERE/c6_pick_rounding.py" --val "$C3/c6val_$1_$2$(sfx "$s")/recs.pkl" 2>/dev/null | tail -1)
    [ "$pick" = cum ] && dir=${dir}_cum
    [ -d "$E/$dir/mfa_corpus" ] && out+="$E/$dir,"
  done
  echo "${out%,}"
}
mirror() {   # 來源資料夾 目的資料夾:複製目錄結構,檔案用 symlink(MFA 的暫存資料夾以 corpus 名稱命名,換名字避免撞名)
  python - "$1" "$2" <<'PY'
import os, sys
from pathlib import Path
src, dst = Path(sys.argv[1]), Path(sys.argv[2])
n = 0
for p in src.rglob("*"):
    if p.suffix in (".wav", ".lab"):
        q = dst / p.relative_to(src)
        q.parent.mkdir(parents=True, exist_ok=True)
        if not q.exists():
            os.symlink(p.resolve(), q)
        n += 1
print(f"  {dst.name}:{n} 個檔案")
PY
}

for d in sixian hailu; do
  X=$O/$d; mkdir -p "$X"
  LEX=$(lex_of "$d"); MODEL=$X/acoustic_trainonly_$d.zip
  [ -f "$LEX" ] || { echo "★ 找不到詞典 $LEX"; continue; }
  act tts
  echo "===== $d:建立只含訓練集 / 測試集的 corpus($(date '+%m-%d %H:%M'))"
  python - "$H" "$d" "$X" <<'PY'
import csv, os, sys
from pathlib import Path
H, d, X = Path(sys.argv[1]), sys.argv[2], Path(sys.argv[3])
idx = {}
for p in (H / "data" / d / "corpus").rglob("*.wav"):
    idx[p.stem] = p
def stems(tsv):
    out = []
    for row in csv.reader(open(tsv, encoding="utf-8"), delimiter="\t"):
        if not row:
            continue
        s = next((Path(c).stem for c in row if c.lower().endswith((".wav", ".flac"))), Path(row[0]).stem)
        if s in idx:
            out.append(s)
    return out
for split, name in (("train_full", f"x3train_{d}"), ("test", f"x3test_{d}")):
    ss = stems(H / "benchmark" / d / f"{split}.tsv")
    dst = X / name
    for s in ss:
        w = idx[s]
        for src in (w, w.with_suffix(".lab")):
            q = dst / w.parent.name / src.name
            q.parent.mkdir(parents=True, exist_ok=True)
            if src.exists() and not q.exists():
                os.symlink(src.resolve(), q)
    print(f"  {split}:{len(ss)} 句 → {dst}")
PY

  act mfa
  if [ ! -f "$MODEL" ]; then
    wait_mfa
    echo "===== $d:mfa train(只用訓練集,-j $JOBS)($(date '+%m-%d %H:%M'))"
    mfa train "$X/x3train_$d" "$LEX" "$MODEL" --clean -j "$JOBS" > "$X/mfa_train.log" 2>&1 \
        || { echo "★ mfa train 失敗,看 $X/mfa_train.log 最後幾行:"; tail -5 "$X/mfa_train.log"; continue; }
  fi
  if ! compgen -G "$X/tg_test/*/*.TextGrid" > /dev/null; then
    wait_mfa
    echo "===== $d:對齊真人測試錄音($(date '+%m-%d %H:%M'))"
    mfa align "$X/x3test_$d" "$LEX" "$MODEL" "$X/tg_test" --clean --single_speaker -j "$JOBS" > "$X/mfa_align_test.log" 2>&1 \
        || { echo "★ 真人對齊失敗,看 $X/mfa_align_test.log"; continue; }
  fi

  act tts
  evs=()
  for k in base c2 c6; do IFS=, read -r -a A <<< "$(dirs_of "$k" "$d")"; evs+=("${A[@]}"); done
  [ -d "$E/big48h_$d/mfa_corpus" ] && evs+=("$E/big48h_$d")
  echo "===== $d:對齊合成語音(${#evs[@]} 個評估資料夾)($(date '+%m-%d %H:%M'))"
  for ev in "${evs[@]}"; do
    [ -n "$ev" ] || continue
    compgen -G "$ev/mfa_tg_x3/*/*.TextGrid" > /dev/null && continue
    syn=$X/x3syn_$(basename "$ev")
    act tts; mirror "$ev/mfa_corpus" "$syn"
    act mfa; wait_mfa
    mfa align "$syn" "$LEX" "$MODEL" "$ev/mfa_tg_x3" --clean --single_speaker -j "$JOBS" > "$X/mfa_align_$(basename "$ev").log" 2>&1 \
        || echo "★ 合成語音對齊失敗:$ev"
  done

  act tts
  echo "===== $d:重新量測($(date '+%m-%d %H:%M'))"
  args=(); for ev in "${evs[@]}"; do [ -n "$ev" ] && args+=(--eval-dir "$ev"); done
  python "$HERE/x3_remeasure.py" --dialect "$d" --nat-tg "$X/tg_test" "${args[@]}"
  sys_args=()
  for k in base c2 c6; do x=$(dirs_of "$k" "$d"); [ -n "$x" ] && sys_args+=(--sys "$k=$x"); done
  [ -d "$E/big48h_$d/mfa_corpus" ] && sys_args+=(--sys "48h=$E/big48h_$d")
  for csvn in syllables_wave.csv syllables_wave_x3.csv; do
    echo "--- $([ "$csvn" = syllables_wave.csv ] && echo '原本的 MFA(全部錄音訓練)' || echo '只用訓練集的 MFA')"
    python "$HERE/x1_overlap.py" --dialect "$d" --csv "$csvn" "${sys_args[@]}" --out "$O/x3_${d}_${csvn%.csv}.md" \
        | sed -n "/## 全部/,/## 排除/p" | grep -v "## 排除"
  done
done
echo "完成($(date '+%m-%d %H:%M'))"
