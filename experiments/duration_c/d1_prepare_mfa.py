"""實驗 D1 第 1 步:把 c_eval.py --synth 合成的音檔整理成 MFA corpus(手冊第 26.2 節)。

用「與參考時長同一套」的客語 MFA 聲學模型與詞典去對齊合成語音,從波形量母音時長,
不再依賴模型內部的 attn 與 blank 分配規則。

    python d1_prepare_mfa.py --eval-dir ~/hakka_tts/exp_c/eval/fair1h_joint_sixian_scratch_s1234__base --dialect sixian
    # 依畫面印出的指令,在 MFA 的 conda 環境執行 mfa align
    python d1_compare.py --eval-dir ~/hakka_tts/exp_c/eval/fair1h_joint_sixian_scratch_s1234__base --dialect sixian

corpus 結構:<eval-dir>/mfa_corpus/<語者>/<檔名>_g<種子>.wav + .lab(.lab 是該句的拼音,取自真人錄音的 TextGrid words 層)
"""
import argparse
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from hakka_c import config as C  # noqa: E402
from hakka_c.textgrid import find_tier, read_textgrid  # noqa: E402


LEXICON = {"sixian": "hakka_lexicon.txt", "hailu": "hailu_lexicon.txt"}


def find_mfa(dialect):
    """你的 TextGrid 是用 `mfa align <corpus> dict/<lexicon> aligned/<腔>/acoustic <out> --single_speaker`
    產生的(~/Documents/MFA/command_history.yaml),聲學模型是一個資料夾。"""
    model = C.HAKKA / "aligned" / dialect / "acoustic"
    lex = C.HAKKA / "dict" / LEXICON[dialect]
    return (lex if lex.exists() else None), (model if (model / "final.mdl").exists() else None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-dir", required=True, help="c_eval.py --synth 的輸出資料夾(含 wav/ 與 bounds/)")
    ap.add_argument("--dialect", required=True, choices=C.DIALECTS)
    ap.add_argument("--out", help="corpus 位置(預設 <eval-dir>/mfa_corpus)")
    args = ap.parse_args()

    ev = Path(args.eval_dir).expanduser()
    wavs = sorted((ev / "wav").glob("*_g*.wav"))
    if not wavs:
        sys.exit(f"★ {ev}/wav 底下沒有合成音檔:先跑 c_eval.py --synth(不要加 --no-wav)")
    out = Path(args.out).expanduser() if args.out else ev / "mfa_corpus"
    if out.exists():
        shutil.rmtree(out)
    tg_idx = {p.stem: p for p in C.textgrid_dir(args.dialect).rglob("*.TextGrid")}
    n, miss = 0, 0
    for w in wavs:
        stem = w.stem.rsplit("_g", 1)[0]
        if stem not in tg_idx:
            miss += 1
            continue
        words = [t.strip() for _, _, t in find_tier(read_textgrid(tg_idx[stem]), "words")
                 if t.strip().lower() not in C.MFA_SILENCE]
        spk = stem.split("-")[0]
        d = out / spk
        d.mkdir(parents=True, exist_ok=True)
        shutil.copy2(w, d / w.name)
        (d / f"{w.stem}.lab").write_text(" ".join(words) + "\n", encoding="utf-8")
        n += 1
    print(f"corpus:{out}({n} 個音檔{f',{miss} 個找不到 TextGrid' if miss else ''})")

    lex, model = find_mfa(args.dialect)
    tg_out = ev / "mfa_tg"
    if not (lex and model):
        print(f"\n★ 找不到詞典或聲學模型(預期 {C.HAKKA}/dict/{LEXICON[args.dialect]} 與 {C.HAKKA}/aligned/{args.dialect}/acoustic/)")
    print("\n請在 MFA 的環境執行(與當初對齊真人錄音的設定相同):")
    print("  conda activate mfa")
    print(f"  mfa align {out} {lex or '<詞典>'} {model or '<聲學模型資料夾>'} {tg_out} --clean --single_speaker")
    print("  conda activate tts")
    print(f"\n完成後:python d1_compare.py --eval-dir {ev} --dialect {args.dialect}")
    json.dump({"corpus": str(out), "tg_out": str(tg_out), "n": n}, open(ev / "d1_corpus.json", "w"), indent=2)


if __name__ == "__main__":
    main()
