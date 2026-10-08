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


def find_candidates():
    roots = [C.HAKKA, Path.home() / "Documents" / "MFA"]
    dicts, models = [], []
    for r in roots:
        if not r.exists():
            continue
        for pat in ("*dict*.txt", "*/*dict*.txt", "*/*/*dict*.txt"):
            dicts += [p for p in r.glob(pat) if "hakka" in p.name.lower() or "mfa" in str(p).lower()]
        for pat in ("*.zip", "*/*.zip", "*/*/*.zip", "*/*/*/*.zip"):
            models += [p for p in r.glob(pat) if p.stat().st_size > 1_000_000]
    return sorted(set(dicts)), sorted(set(models))


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

    dicts, models = find_candidates()
    tg_out = ev / "mfa_tg"
    print("\n找到的詞典候選:", [str(p) for p in dicts][:6] or "(沒找到,請自己指定 build_mfa_dict.py 產生的 hakka_dict.txt)")
    print("找到的聲學模型候選(.zip):", [str(p) for p in models][:6] or "(沒找到;mfa train 輸出的模型 .zip)")
    print("\n請在 MFA 的環境執行(詞典與聲學模型要用當初對齊真人錄音的那一套):")
    print(f"  mfa align {out} <hakka_dict.txt> <客語聲學模型.zip> {tg_out} --clean --beam 100 --retry_beam 400")
    print(f"\n完成後:python d1_compare.py --eval-dir {ev} --dialect {args.dialect}")
    json.dump({"corpus": str(out), "tg_out": str(tg_out), "n": n}, open(ev / "d1_corpus.json", "w"), indent=2)


if __name__ == "__main__":
    main()
