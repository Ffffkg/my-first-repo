"""實驗 C0:入聲母音偏長的來源診斷(手冊第 25.3 節)。不需要重訓,也不跑解碼器。

對每一句真人錄音同時取得四種母音核時長:
    ref   MFA TextGrid(參考)
    mas   模型在真人錄音上的 teacher-forced MAS(= 訓練時時長預測器的學習目標)
    cont  時長預測器的連續輸出 w = exp(logw)
    ceil  Matcha synthesise() 實際使用的 ceil(w)(= 論文主表 ET-VBias 量的東西)
    cum   改用累積四捨五入(總和守恆)的 w
每一種都用四種 blank 分配規則(half / left / right / none)換算,寫成逐音節 CSV,
並把逐句原始陣列存成 pickle,給 c0_report.py / c1_calibrate.py 用(不需要 GPU)。

範例:
    # 先做回歸核對:48h baseline、測試集前 100 句,應重現主表(四縣 ET-VBias +23.3、NonET +0.4)
    python c0_diag.py --exp hakka_joint_sixian_ladder_48h --dialect sixian --first 100
    # 正式:全部測試句 + 訓練集抽 1000 句(看訓練目標本身)
    python c0_diag.py --exp hakka_joint_sixian_ladder_48h --dialect sixian
    python c0_diag.py --exp hakka_joint_sixian_ladder_48h --dialect sixian --split train_full --max-utts 1000
"""
import argparse
import pickle
import random
import sys
import time
from collections import Counter
from dataclasses import asdict
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parent))
from hakka_c import config as C  # noqa: E402
from hakka_c.analysis import base_row  # noqa: E402
from hakka_c.csvio import write_rows  # noqa: E402
from hakka_c.durations import RULES, matcha_frames, merge_blank  # noqa: E402
from hakka_c.matcha_io import Engine, find_ckpt, load_mel_stats, read_filelist, textgrid_index  # noqa: E402
from hakka_c.structure import StructError, build_utt, is_pause  # noqa: E402
from hakka_c.textgrid import read_textgrid  # noqa: E402


def pick_rows(rows, args):
    if args.first:
        return sorted(rows, key=lambda r: Path(r["wav"]).stem)[: args.first]
    if args.max_utts and len(rows) > args.max_utts:
        rng = random.Random(C.SPLIT_SEED)
        return rng.sample(rows, args.max_utts)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", help="實驗名稱,例如 hakka_joint_sixian_ladder_48h")
    ap.add_argument("--ckpt", help="直接指定 checkpoint(優先於 --exp)")
    ap.add_argument("--which", default="best", help="best / last / step40000")
    ap.add_argument("--dialect", required=True, choices=C.DIALECTS)
    ap.add_argument("--mode", default="joint", choices=("joint", "separate"))
    ap.add_argument("--split", default="test", help="filelist 名稱:test / val / train_full / ladder_1h ...")
    ap.add_argument("--filelist", help="直接指定 filelist(優先於 --split)")
    ap.add_argument("--first", type=int, default=0, help="依檔名排序取前 N 句(重現主表用 100)")
    ap.add_argument("--max-utts", type=int, default=0, help="隨機抽 N 句(種子 20260716)")
    ap.add_argument("--data-yaml", help="用這個 data config 的 mel 統計量(遷移模型必填;從零模型可省略)")
    ap.add_argument("--cleaners", default="hakka_cleaners")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--tag", default="", help="輸出檔名後綴")
    args = ap.parse_args()

    ckpt = Path(args.ckpt) if args.ckpt else find_ckpt(args.exp, args.which)
    cond = args.exp or ckpt.parent.parent.parent.parent.name
    fl = Path(args.filelist) if args.filelist else C.filelist(args.mode, args.dialect, args.split)
    rows = pick_rows(read_filelist(fl), args)
    protocol = f"first{args.first}" if args.first else ("sample" if args.max_utts else "all")
    print(f"checkpoint: {ckpt}\nfilelist:   {fl}({len(rows)} 句,{protocol})")

    eng = Engine(ckpt, args.device, (args.cleaners,), load_mel_stats(args.data_yaml) if args.data_yaml else None)
    print(f"mel 統計量:{eng.stats_src}({eng.mel_mean:.4f}, {eng.mel_std:.4f})")
    tg_idx = textgrid_index(args.dialect)

    out_rows, recs, skipped = [], [], Counter()
    t0 = time.time()
    for j, r in enumerate(rows):
        stem = Path(r["wav"]).stem
        try:
            x, names = eng.text(r["text"])
            if stem not in tg_idx:
                raise StructError("找不到 TextGrid")
            utt = build_utt(stem, names, read_textgrid(tg_idx[stem]))
            if x.shape[1] != 2 * len(names) + 1:
                raise StructError("符號長度不是 2n+1")
        except (StructError, KeyError) as e:
            skipped[str(e)[:60]] += 1
            continue
        mu_x, x_mask, _, w = eng.encode(x, r["spk"])
        y = eng.mel(r["wav"])
        mas = eng.mas(mu_x, x_mask, y)
        T = int(y.shape[-1])
        assert mas.sum() == T, (stem, mas.sum(), T)

        src = {"mas": mas.astype(float), "cont": w, "ceil": matcha_frames(w, mode="ceil"),
               "cum": matcha_frames(w, mode="cum")}
        merged = {(s, rule): merge_blank(v, rule) * C.HOP_MS for s, v in src.items() for rule in RULES}
        # 音素層級的總長比(只算音素,不含停頓與頭尾靜音;blank 依 half 規則分回音素)
        mfa_ms = sum(b - a for a, b in utt.phone_iv) * 1000
        phone_ratio = float(merged[("ceil", "half")][utt.phone_sym].sum()) / mfa_ms
        rec = {"dialect": args.dialect, "stem": stem, "spk": r["spk"], "T": T, "w": w, "mas": mas,
               "n": len(names), "names": names, "pause": [i for i, nm in enumerate(names) if is_pause(nm)],
               "syllables": [asdict(s) for s in utt.syllables]}
        recs.append(rec)
        for s in rec["syllables"]:
            row = {"cond": cond, "protocol": protocol, **base_row(rec, s)}
            for (sname, rule), v in merged.items():
                row[f"{sname}_{rule}_ms"] = round(float(v[s["nuc"]]), 3)
            row["utt_T"] = T
            row["utt_ceil"] = int(src["ceil"].sum())
            row["utt_cont"] = round(float(w.sum()), 2)
            row["utt_phone_ratio"] = round(phone_ratio, 4)
            out_rows.append(row)
        if (j + 1) % 50 == 0:
            print(f"  {j + 1}/{len(rows)}  {time.time() - t0:.0f}s")

    tag = f"_{args.tag}" if args.tag else ""
    base = C.OUT_ROOT / "c0" / f"{cond}_{args.split}_{protocol}{tag}"
    write_rows(f"{base}.csv", out_rows)
    pickle.dump({"cond": cond, "ckpt": str(ckpt), "protocol": protocol, "recs": recs},
                open(f"{base}.pkl", "wb"))
    print(f"\n完成 {len(recs)}/{len(rows)} 句、{len(out_rows)} 個音節 → {base}.csv / .pkl")
    if skipped:
        print("跳過:", dict(skipped.most_common()))
    ratio = sum(matcha_frames(r["w"]).sum() for r in recs) / max(1, sum(r["T"] for r in recs))
    pr = [r["utt_phone_ratio"] for r in out_rows]
    print(f"合成/真實總長比:整句含靜音 {ratio:.3f};只算音素 ≈ {sum(pr) / max(len(pr), 1):.3f}")
    print(f"接著執行:python c0_report.py {base}.csv")


if __name__ == "__main__":
    main()
