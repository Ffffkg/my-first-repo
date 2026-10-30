"""C3 前置診斷:C2 的母音核時長偏差,拆成「訓練目標 / 預測器 / 取整 / blank 裡的靜音」各貢獻多少。

C2 是拿 MFA 轉出的 durations/<stem>.npy 當訓練目標。對每個音節同時取:
    ref     MFA TextGrid 的母音核時長(c_eval 的參考值)
    tgt     .npy 訓練目標,merge_blank(half) 後的母音核
    cont    時長預測器的連續輸出 w,merge_blank(half)
    ceil    synthesise() 實際用的 ceil(w),merge_blank(half)(= c_eval 的 pred_ms)
並拆成
    A 目標 − 參考   = tgt − ref      (MFA → frame 的轉換誤差,包含「blank 裡的靜音」被 half 規則分給母音)
    S 其中的靜音    = 目標中相鄰 blank 超過 1 frame 的部分,依 half 規則分給母音核的量
    B 預測 − 目標   = cont − tgt     (預測器本身)
    R 取整          = ceil − cont
    合計            = ceil − ref     (= c_eval 報的 VBias)
    N 只看母音核    = ceil(w)[母音核] − 目標[母音核](不含任何 blank,最乾淨的「預測器有沒有把母音唸長」)

    python c3_target_check.py --exp c2_mfadur_fair1h_joint_sixian_scratch_s1234 --dialect sixian --which step40000 \
        --filelist ~/Matcha-TTS/data/hakka/joint/sixian/ladder_1h_mfadur.txt --name c2_sixian_train
    python c3_target_check.py --exp c2_mfadur_fair1h_joint_sixian_scratch_s1234 --dialect sixian --which step40000 \
        --filelist ~/Matcha-TTS/data/hakka/joint/sixian/test_mfadur.txt --name c2_sixian_test

只跑編碼器(不合成),GPU 用量很小。
"""
import argparse
import pickle
import sys
import time
from collections import Counter
from dataclasses import asdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from hakka_c import config as C  # noqa: E402
from hakka_c.analysis import base_row  # noqa: E402
from hakka_c.csvio import write_rows  # noqa: E402
from hakka_c.durations import matcha_frames, merge_blank  # noqa: E402
from hakka_c.stats import UttTable, boot_idx, fmt, mean_ci  # noqa: E402

COMPONENTS = (("A", "目標−參考"), ("S", "其中:blank 靜音"), ("B", "預測−目標"), ("R", "取整"),
              ("total", "合計 ceil−參考"), ("N", "只看母音核"))


def syllable_components(d, w, syllables):
    """d:訓練目標 frame 數(2n+1);w:預測器連續輸出(2n+1);syllables:Syllable 的 dict。回傳每音節的分量(ms)。"""
    d = np.asarray(d, float)
    w = np.asarray(w, float)
    ceil = matcha_frames(w, mode="ceil").astype(float)
    d_cap = d.copy()
    d_cap[0::2] = np.minimum(d_cap[0::2], 1.0)          # blank 只留 1 frame = 不含靜音的目標
    tgt, tgt_cap = merge_blank(d, "half"), merge_blank(d_cap, "half")
    cont, cei = merge_blank(w, "half"), merge_blank(ceil, "half")
    out = []
    for s in syllables:
        i, p = s["nuc"], 2 * s["nuc"] + 1
        ms = {"ref": s["ref_ms"], "tgt": tgt[i] * C.HOP_MS, "cont": cont[i] * C.HOP_MS, "ceil": cei[i] * C.HOP_MS}
        out.append({
            **{f"{k}_ms": round(float(v), 3) for k, v in ms.items()},
            "A": round(float(ms["tgt"] - ms["ref"]), 3),
            "S": round(float((tgt[i] - tgt_cap[i]) * C.HOP_MS), 3),
            "B": round(float(ms["cont"] - ms["tgt"]), 3),
            "R": round(float(ms["ceil"] - ms["cont"]), 3),
            "total": round(float(ms["ceil"] - ms["ref"]), 3),
            "N": round(float((ceil[p] - d[p]) * C.HOP_MS), 3),
            "open": int(s["coda"] == ""),
            "blank_r_tgt": int(d[p + 1]), "blank_r_pred": int(ceil[p + 1]),
        })
    return out


def report(rows, n_boot=2000):
    stem = np.array([r["stem"] for r in rows])
    et = np.array([r["entering"] for r in rows], bool)
    pp = np.array([r["prepausal"] for r in rows], bool)
    op = np.array([r["open"] for r in rows], bool)
    fem = np.array([r["gender"] == "F" for r in rows], bool)
    groups = {
        "入聲": et, "舒聲": ~et,
        "入聲・非停頓前": et & ~pp, "入聲・停頓前": et & pp,
        "舒聲・非停頓前": ~et & ~pp, "舒聲・停頓前": ~et & pp,
        "舒聲・開音節(無韻尾)": ~et & op, "舒聲・有鼻音韻尾": ~et & ~op,
        "入聲・女": et & fem, "入聲・男": et & ~fem, "舒聲・女": ~et & fem, "舒聲・男": ~et & ~fem,
    }
    groups = {k: m for k, m in groups.items() if m.sum() >= 15}
    t = UttTable(stem, {c: [r[c] for r in rows] for c, _ in COMPONENTS}, groups)
    idx = boot_idx(t.U, n_boot)
    L = ["| 分組 | n | " + " | ".join(lab for _, lab in COMPONENTS) + " |",
         "|---|---|" + "---|" * len(COMPONENTS)]
    for g, m in groups.items():
        cells = [fmt(mean_ci(t, c, g, idx)) if c in ("total", "N", "B") else f"{t.mean(c, g):+.1f}"
                 for c, _ in COMPONENTS]
        L.append(f"| {g} | {int(m.sum())} | " + " | ".join(cells) + " |")
    br_t = np.array([r["blank_r_tgt"] for r in rows])
    br_p = np.array([r["blank_r_pred"] for r in rows])
    L += ["", "母音核右邊那個 blank 的 frame 數(目標 / 預測 ceil),平均:",
          f"  舒聲開音節 {br_t[~et & op].mean():.2f} / {br_p[~et & op].mean():.2f};"
          f"舒聲停頓前 {br_t[~et & pp].mean():.2f} / {br_p[~et & pp].mean():.2f};"
          f"其他 {br_t[~(~et & (op | pp))].mean():.2f} / {br_p[~(~et & (op | pp))].mean():.2f}"]
    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp")
    ap.add_argument("--ckpt")
    ap.add_argument("--which", default="step40000")
    ap.add_argument("--dialect", required=True, choices=C.DIALECTS)
    ap.add_argument("--filelist", required=True, help="*_mfadur.txt(只有它們保證有 .npy)")
    ap.add_argument("--dur-dir", help="預設 <wav>/../../durations")
    ap.add_argument("--name", required=True)
    ap.add_argument("--cleaners", default="hakka_cleaners")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--n-boot", type=int, default=2000)
    args = ap.parse_args()

    from hakka_c.matcha_io import Engine, find_ckpt, read_filelist, textgrid_index
    from hakka_c.structure import StructError, build_utt
    from hakka_c.textgrid import read_textgrid

    ckpt = Path(args.ckpt) if args.ckpt else find_ckpt(args.exp, args.which)
    rows_in = read_filelist(Path(args.filelist).expanduser())
    print(f"checkpoint {ckpt}\nfilelist {args.filelist}({len(rows_in)} 句)")
    eng = Engine(ckpt, args.device, (args.cleaners,))
    tg_idx = textgrid_index(args.dialect)

    rows, recs, skipped, t0 = [], [], Counter(), time.time()
    for j, r in enumerate(rows_in):
        stem = Path(r["wav"]).stem
        dur_dir = Path(args.dur_dir).expanduser() if args.dur_dir else Path(r["wav"]).parent.parent / "durations"
        try:
            x, names = eng.text(r["text"])
            if stem not in tg_idx:
                raise StructError("找不到 TextGrid")
            if not (dur_dir / f"{stem}.npy").exists():
                raise StructError("找不到 durations .npy")
            d = np.load(dur_dir / f"{stem}.npy")
            if len(d) != 2 * len(names) + 1:
                raise StructError(".npy 長度不是 2n+1")
            utt = build_utt(stem, names, read_textgrid(tg_idx[stem]))
        except (StructError, KeyError) as e:
            skipped[str(e)[:60]] += 1
            continue
        _, _, _, w = eng.encode(x, r["spk"])
        rec = {"dialect": args.dialect, "stem": stem, "spk": r["spk"], "names": names}
        syls = [asdict(s) for s in utt.syllables]
        recs.append({**rec, "d": d.astype(int), "w": np.asarray(w, float), "syllables": syls})
        for s, comp in zip(syls, syllable_components(d, w, syls)):
            rows.append({**base_row(rec, s), **comp})
        if (j + 1) % 100 == 0:
            print(f"  {j + 1}/{len(rows_in)}  {time.time() - t0:.0f}s")

    out = C.OUT_ROOT / "c3" / args.name
    write_rows(out / "components.csv", rows)
    pickle.dump({"ckpt": str(ckpt), "filelist": str(args.filelist), "recs": recs}, open(out / "recs.pkl", "wb"))
    text = (f"# C3 前置診斷:{args.name}\n\ncheckpoint `{ckpt}`,{len({r['stem'] for r in rows})} 句,"
            f"{len(rows)} 個音節。單位 ms;合計、只看母音核、預測−目標附句子層級 bootstrap 95% CI。\n\n"
            + report(rows, args.n_boot)
            + "\n\n讀法:合計 = 目標−參考 + 預測−目標 + 取整。「只看母音核」不含 blank,"
              "是預測器有沒有把母音本身唸長的最直接指標;「blank 靜音」大,代表 half 規則把停頓算進了母音(量法問題)。")
    (out / "report.md").write_text(text, encoding="utf-8")
    print("\n" + text)
    if skipped:
        print("\n跳過:", dict(skipped.most_common()))
    print(f"\n→ {out}/components.csv、report.md、recs.pkl(給 c3_fit_pp.py)")


if __name__ == "__main__":
    main()
