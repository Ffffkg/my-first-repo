"""G1 第 1 步:抽每個音節母音核的 F0 特徵(真人錄音或合成語音),存成 CSV。不需要 GPU。

真人錄音(分類器的訓練資料與「天花板」):TextGrid 是原始音檔座標,22 kHz 音檔有裁切,
用 margin_offset 重現裁切起點(與 C2 相同);對不上的句子跳過。
    python g1_tone_feats.py real --dialect sixian --split train_full --max-utts 3000
    python g1_tone_feats.py real --dialect sixian --split test

合成語音:用 D1 已經做好的 MFA 對齊(<eval-dir>/mfa_tg)與 bounds,每個生成種子都算。
    python g1_tone_feats.py synth --dialect sixian --eval-dir ~/hakka_tts/exp_c/eval/c2_sixian_step40000

輸出欄位:stem, spk, gen_seed(真人 = -1), syl_idx, word, tone, entering, coda, prepausal, dur_ms, vfrac, p1..p5(log2 Hz)
"""
import argparse
import json
import random
import sys
from collections import Counter
from dataclasses import asdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from hakka_c import config as C  # noqa: E402
from hakka_c.csvio import write_rows  # noqa: E402
from hakka_c.durations import margin_offset  # noqa: E402
from hakka_c.f0feat import f0_track, nucleus_points  # noqa: E402
from hakka_c.structure import StructError, build_utt  # noqa: E402
from hakka_c.textgrid import find_tier, read_textgrid  # noqa: E402

SPK_ID = {"XF": 0, "XM": 1, "HF": 2, "HM": 3}
META = ("word", "tone", "entering", "coda", "prepausal")


def syllable_rows(utt, syls, f0, t, offset, stem, spk, gen_seed):
    """utt:對齊結構(提供母音核區間);syls:要輸出的音節 dict(提供聲調等標記,依 idx 對應)。"""
    iv = dict(zip(utt.phone_sym, utt.phone_iv))
    by_idx = {s["idx"]: s for s in syls}
    rows = []
    for s in utt.syllables:
        if s.idx not in by_idx or s.nuc not in iv:
            continue
        a, b = iv[s.nuc]
        t0, t1 = a - offset, b - offset
        pts, vfrac = nucleus_points(f0, t, t0, t1)
        m = by_idx[s.idx]
        rows.append({"stem": stem, "spk": spk, "gen_seed": gen_seed, "syl_idx": s.idx,
                     **{k: (int(m[k]) if isinstance(m[k], bool) else m[k]) for k in META},
                     "dur_ms": round((t1 - t0) * 1000, 2), "vfrac": round(vfrac, 3),
                     **{f"p{k + 1}": (round(v, 5) if np.isfinite(v) else "") for k, v in enumerate(pts)}})
    return rows


def run_real(args, sf):
    sys.path.insert(0, str(C.MATCHA))
    from matcha.text import text_to_sequence
    from hakka_c.matcha_io import read_filelist, textgrid_index
    from hakka_c.pua import resolve_pua_map

    pua, _ = resolve_pua_map((args.cleaners,), verbose=False)
    fl = Path(args.filelist).expanduser() if args.filelist else C.filelist(args.mode, args.dialect, args.split)
    rows_in = read_filelist(fl)
    if args.max_utts and len(rows_in) > args.max_utts:
        rows_in = random.Random(C.SPLIT_SEED).sample(rows_in, args.max_utts)
    tg_idx = textgrid_index(args.dialect)
    out, why = [], Counter()
    for j, r in enumerate(rows_in):
        stem = Path(r["wav"]).stem
        try:
            _, clean = text_to_sequence(r["text"], [args.cleaners])
            names = [pua[c] for c in clean]
            if stem not in tg_idx:
                raise StructError("找不到 TextGrid")
            tiers = read_textgrid(tg_idx[stem])
            utt = build_utt(stem, names, tiers)
            wav, sr = sf.read(r["wav"])
            if wav.ndim > 1:
                wav = wav.mean(axis=1)
            orig_dur = max(b for _, b, _ in find_tier(tiers, "phones"))
            off = margin_offset(utt.phone_iv[0][0], utt.phone_iv[-1][1], orig_dur, len(wav) / sr,
                                args.margin_ms / 1000, args.tail_tol_ms / 1000)
            if off is None:
                raise StructError("裁切起點對不上")
        except (StructError, KeyError, ValueError, FileNotFoundError) as e:
            why[str(e)[:50]] += 1
            continue
        f0, t = f0_track(wav, sr)
        out += syllable_rows(utt, [asdict(s) for s in utt.syllables], f0, t, off, stem, r["spk"], -1)
        if (j + 1) % 200 == 0:
            print(f"  {j + 1}/{len(rows_in)}")
    dest = Path(args.out).expanduser() if args.out else C.OUT_ROOT / "g1" / f"real_{args.dialect}_{args.split}.csv"
    return out, why, dest


def run_synth(args, sf):
    ev = Path(args.eval_dir).expanduser()
    tgs = {p.stem: p for p in (ev / "mfa_tg").rglob("*.TextGrid")}
    if not tgs:
        sys.exit(f"★ {ev}/mfa_tg 沒有 TextGrid:先跑 d1_pipeline.sh(或 D1 的 mfa align)")
    out, why = [], Counter()
    for name, tgp in sorted(tgs.items()):
        stem, seed = name.rsplit("_g", 1)
        bj, wp = ev / "bounds" / f"{stem}.json", ev / "wav" / f"{name}.wav"
        if not bj.exists() or not wp.exists():
            why["沒有 bounds 或 wav"] += 1
            continue
        b = json.load(open(bj, encoding="utf-8"))
        try:
            utt = build_utt(stem, b["names"], read_textgrid(tgp))
        except (StructError, KeyError) as e:
            why[str(e)[:50]] += 1
            continue
        wav, sr = sf.read(wp)
        f0, t = f0_track(wav, sr)
        out += syllable_rows(utt, b["syllables"], f0, t, 0.0, stem, SPK_ID.get(stem.split("-")[0], -1), int(seed))
    dest = Path(args.out).expanduser() if args.out else ev / "syl_f0.csv"
    return out, why, dest


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("source", choices=("real", "synth"))
    ap.add_argument("--dialect", required=True, choices=C.DIALECTS)
    ap.add_argument("--eval-dir", help="synth:d1_pipeline.sh 的輸出資料夾")
    ap.add_argument("--split", default="test", help="real:filelist 名稱(test / train_full / ladder_10h …)")
    ap.add_argument("--filelist", help="real:直接指定 filelist")
    ap.add_argument("--mode", default="joint", choices=("joint", "separate"))
    ap.add_argument("--max-utts", type=int, default=0, help="real:隨機抽 N 句(種子 20260716)")
    ap.add_argument("--margin-ms", type=float, default=50, help="resample_audio.py 的 MARGIN(與 C2 相同)")
    ap.add_argument("--tail-tol-ms", type=float, default=25)
    ap.add_argument("--cleaners", default="hakka_cleaners")
    ap.add_argument("--out")
    args = ap.parse_args()

    import soundfile as sf
    if args.source == "synth" and not args.eval_dir:
        sys.exit("★ synth 需要 --eval-dir")
    out, why, dest = (run_real if args.source == "real" else run_synth)(args, sf)
    if not out:
        sys.exit(f"★ 沒有任何音節:{dict(why)}")
    write_rows(dest, out)
    n_utt = len({(r["stem"], r["gen_seed"]) for r in out})
    print(f"→ {dest}:{n_utt} 句(× 生成種子)、{len(out)} 個音節;跳過 {dict(why) or '無'}")


if __name__ == "__main__":
    main()
