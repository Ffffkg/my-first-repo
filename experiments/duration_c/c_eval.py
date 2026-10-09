"""評估任一個模型(baseline、C2、C3…)的入聲時長指標,可選擇同時合成算 MCD。

ET 指標:時長預測器是確定性的(推論時沒有 dropout),只跑編碼器就能得到合成時長,
所以用「全部測試句」計算,不受生成種子影響。
MCD(--synth):依手冊第 24.2 節協議,每語者 50 句 × 5 個生成種子;同時存下 wav 與
每個符號的邊界(秒),實驗 D 會直接拿來做波形量測。

    # baseline(Matcha 原本的 ceil)
    python c_eval.py --exp hakka_joint_sixian_ladder_48h --dialect sixian
    # C1 校正後合成(參數來自 c1_calibrate.py)
    python c_eval.py --exp hakka_joint_sixian_ladder_48h --dialect sixian \
        --calib ~/hakka_tts/exp_c/c1/hakka_joint_sixian_ladder_48h/calib.json --variant class_cum --synth
    # C2 模型
    python c_eval.py --exp c2_mfadur_joint_sixian_1h --dialect sixian --synth
"""
import argparse
import json
import pickle
import random
import sys
import time
from collections import Counter, defaultdict
from dataclasses import asdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from hakka_c import config as C  # noqa: E402
from hakka_c.analysis import eval_rows, pred_frames, summarize  # noqa: E402
from hakka_c.csvio import read_cols, write_rows  # noqa: E402
from hakka_c.matcha_io import Engine, find_ckpt, import_eval_v2, load_mel_stats, read_filelist, textgrid_index  # noqa: E402
from hakka_c.structure import StructError, build_utt, is_pause  # noqa: E402
from hakka_c.textgrid import read_textgrid  # noqa: E402


def synth_list(rows, per_spk, exclude):
    """每位語者依檔名排序後,用種子 20260716 抽 per_spk 句(排除 exclude 中的檔名)。"""
    by = defaultdict(list)
    for r in sorted(rows, key=lambda r: Path(r["wav"]).stem):
        if Path(r["wav"]).stem not in exclude:
            by[r["spk"]].append(r)
    rng = random.Random(C.SPLIT_SEED)
    out = []
    for spk in sorted(by):
        pool = by[spk]
        out += sorted(rng.sample(pool, min(per_spk, len(pool))), key=lambda r: Path(r["wav"]).stem)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp")
    ap.add_argument("--ckpt")
    ap.add_argument("--which", default="best")
    ap.add_argument("--dialect", required=True, choices=C.DIALECTS)
    ap.add_argument("--mode", default="joint", choices=("joint", "separate"))
    ap.add_argument("--split", default="test")
    ap.add_argument("--filelist")
    ap.add_argument("--calib", help="c1_calibrate.py 的 calib.json")
    ap.add_argument("--variant", default="base", help="calib.json 中的變體名稱")
    ap.add_argument("--rounding", choices=("ceil", "cum"))
    ap.add_argument("--length-scale", type=float)
    ap.add_argument("--s-et", type=float)
    ap.add_argument("--s-non", type=float)
    ap.add_argument("--data-yaml", help="mel 統計量來源(遷移模型必填)")
    ap.add_argument("--cleaners", default="hakka_cleaners")
    ap.add_argument("--synth", action="store_true", help="合成並計算 MCD,存 wav 與邊界")
    ap.add_argument("--per-spk", type=int, default=50)
    ap.add_argument("--gen-seeds", default=",".join(map(str, C.GEN_SEEDS)))
    ap.add_argument("--exclude-stems", help="每行一個檔名(例如跨腔重疊句)")
    ap.add_argument("--no-wav", action="store_true", help="不存 wav(省空間;實驗 D 需要 wav)")
    ap.add_argument("--vocoder", help="HiFi-GAN universal v1 checkpoint(預設 Matcha 的下載位置)")
    ap.add_argument("--selfcheck", action="store_true", help="先驗證自寫合成路徑與 model.synthesise() 一致")
    ap.add_argument("--n-boot", type=int, default=10000)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--name", help="輸出資料夾名稱(預設 <cond>__<variant>)")
    args = ap.parse_args()

    keys = ("rounding", "length_scale", "s_et", "s_non")
    p = {"rounding": "ceil", "length_scale": 1.0, "s_et": 1.0, "s_non": 1.0}
    label = "base"
    if args.calib:
        p.update(json.load(open(args.calib, encoding="utf-8"))[args.variant])
        label = args.variant
    overrides = {k: getattr(args, k) for k in keys if getattr(args, k) is not None}
    if overrides:
        p.update(overrides)
        label += "_custom"
    kw = {k: p[k] for k in keys}

    ckpt = Path(args.ckpt) if args.ckpt else find_ckpt(args.exp, args.which)
    cond = args.exp or ckpt.parent.parent.parent.parent.name
    name = args.name or f"{cond}__{label}"
    out = C.OUT_ROOT / "eval" / name
    out.mkdir(parents=True, exist_ok=True)
    fl = Path(args.filelist) if args.filelist else C.filelist(args.mode, args.dialect, args.split)
    rows = read_filelist(fl)
    print(f"checkpoint {ckpt}\n推論設定 {kw}\n輸出 {out}")

    eng = Engine(ckpt, args.device, (args.cleaners,), load_mel_stats(args.data_yaml) if args.data_yaml else None)
    if args.selfcheck:
        ok, msg = eng.selfcheck(rows[0]["text"], rows[0]["spk"])
        print(f"自我檢查:{'一致 ✓' if ok else '★不一致'}({msg})")
        if not ok:
            sys.exit(1)
    tg_idx = textgrid_index(args.dialect)

    # ---- ET 指標(全部測試句,只跑編碼器)
    recs, skipped = [], Counter()
    for r in rows:
        stem = Path(r["wav"]).stem
        try:
            x, names = eng.text(r["text"])
            if stem not in tg_idx:
                raise StructError("找不到 TextGrid")
            utt = build_utt(stem, names, read_textgrid(tg_idx[stem]))
        except (StructError, KeyError) as e:
            skipped[str(e)[:60]] += 1
            continue
        mu_x, x_mask, spks, w = eng.encode(x, r["spk"])
        recs.append({"dialect": args.dialect, "stem": stem, "spk": r["spk"], "T": None, "w": w, "mas": None,
                     "n": len(names), "names": names, "pause": [i for i, nm in enumerate(names) if is_pause(nm)],
                     "syllables": [asdict(s) for s in utt.syllables], "wav": r["wav"], "text": r["text"]})
    syl_rows = eval_rows(recs, **kw)
    for sr in syl_rows:
        sr["cond"], sr["variant"] = cond, name
    write_rows(out / "syllables.csv", syl_rows)
    pickle.dump({"cond": cond, "ckpt": str(ckpt), "params": kw, "recs": recs}, open(out / "recs.pkl", "wb"))
    summ, txt = summarize(read_cols(out / "syllables.csv"), n_boot=args.n_boot)
    print(f"\n入聲時長指標({len(recs)}/{len(rows)} 句):\n{txt}")
    if skipped:
        print("跳過:", dict(skipped))
    json.dump({"cond": cond, "ckpt": str(ckpt), "params": kw,
               **{k: (list(v) if isinstance(v, tuple) else v) for k, v in summ.items()}},
              open(out / "summary.json", "w"), indent=2, ensure_ascii=False)

    if not args.synth:
        print(f"\n→ {out}/syllables.csv、summary.json(加 --synth 才會算 MCD)")
        return

    # ---- 合成 + MCD
    ev = import_eval_v2()
    exclude = set(l.strip() for l in open(args.exclude_stems, encoding="utf-8")) if args.exclude_stems else set()
    by_stem = {r["stem"]: r for r in recs}
    todo = [r for r in synth_list(rows, args.per_spk, exclude) if Path(r["wav"]).stem in by_stem]
    seeds = [int(s) for s in args.gen_seeds.split(",")]
    eng.load_vocoder(args.vocoder)
    (out / "wav").mkdir(exist_ok=True)
    (out / "bounds").mkdir(exist_ok=True)
    import soundfile as sf
    mcd_rows, f0_ok, t0 = [], True, time.time()
    for j, r in enumerate(todo):
        rec = by_stem[Path(r["wav"]).stem]
        frames = pred_frames(rec, **kw)
        x, _ = eng.text(rec["text"])
        mu_x, x_mask, spks, _ = eng.encode(x, rec["spk"])
        ref = eng.wav(rec["wav"]).astype(np.float64)
        f0_ref, mc_ref = ev.analyze_wav(ref)
        edges = np.r_[0, np.cumsum(frames)] * C.HOP / C.SR
        json.dump({"stem": rec["stem"], "names": rec["names"], "frames": frames.tolist(),
                   "start_s": edges[:-1].tolist(), "end_s": edges[1:].tolist(),
                   "syllables": rec["syllables"], "params": kw},
                  open(out / "bounds" / f"{rec['stem']}.json", "w"), ensure_ascii=False)
        for gs in seeds:
            dec, _ = eng.decode(mu_x, x_mask, spks, frames, gs)
            syn = eng.to_wav(dec).astype(np.float64)
            assert len(syn) == int(frames.sum()) * C.HOP, (len(syn), frames.sum())
            if not args.no_wav:
                sf.write(out / "wav" / f"{rec['stem']}_g{gs}.wav", syn, C.SR)
            f0_syn, mc_syn = ev.analyze_wav(syn)
            row = {"stem": rec["stem"], "spk": rec["spk"], "gender": C.SPK_GENDER.get(rec["spk"], "?"),
                   "gen_seed": gs, "mcd": float(ev.mcd_from_mcep(mc_syn, mc_ref))}
            if f0_ok and hasattr(ev, "f0_metrics"):
                try:
                    m = ev.f0_metrics(f0_syn, f0_ref)
                    if isinstance(m, (tuple, list)) and len(m) >= 2:
                        row["f0_rmse"], row["vde"] = float(m[0]), float(m[1])
                    elif isinstance(m, dict):
                        row.update({k: float(v) for k, v in m.items() if np.isscalar(v)})
                except Exception as e:  # noqa: BLE001
                    print(f"  f0_metrics 呼叫失敗,之後只算 MCD:{e}")
                    f0_ok = False
            mcd_rows.append(row)
        if (j + 1) % 10 == 0:
            print(f"  合成 {j + 1}/{len(todo)}  {time.time() - t0:.0f}s")
    write_rows(out / "mcd.csv", mcd_rows)
    m = np.array([r["mcd"] for r in mcd_rows])
    g = np.array([r["gender"] for r in mcd_rows])
    print(f"\nMCD 平均 {m.mean():.3f} dB(女 {m[g == 'F'].mean():.3f}、男 {m[g == 'M'].mean():.3f}),"
          f"{len(todo)} 句 × {len(seeds)} 種子")
    print(f"→ {out}/syllables.csv、mcd.csv、wav/、bounds/")


if __name__ == "__main__":
    main()
