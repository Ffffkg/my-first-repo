"""X3 第 2 步:用「只在訓練集上訓練」的 MFA 重新量測母音核時長。不需要 GPU。

論文主實驗的 MFA 是用全部錄音(含驗證、測試句)訓練的。這裡改用只看過 train_full 的 MFA,
重新對齊真人測試錄音(參考時長)與合成語音(預測時長),逐音節重算,輸出與 syllables_wave.csv
同欄位的 syllables_wave_x3.csv,讓 x1_overlap.py --csv syllables_wave_x3.csv 出同一張表。
音節集合沿用原本 D1 的 bounds(同一批句子、同一批音節),只換掉兩邊的時長。

    python x3_remeasure.py --dialect sixian --nat-tg ~/hakka_tts/exp_c/x3/sixian/tg_test \\
        --eval-dir ~/hakka_tts/exp_c/eval/c6_sixian_step40000_cum --synth-tg <同上>/mfa_tg_x3
"""
import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from hakka_c import config as C  # noqa: E402
from hakka_c.csvio import write_rows  # noqa: E402
from hakka_c.stats import pearson  # noqa: E402
from hakka_c.structure import StructError, build_utt  # noqa: E402
from hakka_c.textgrid import read_textgrid  # noqa: E402


def nucleus_ms(stem, names, tg_path):
    """{音節序號: 母音核時長 ms};對不上時回傳 None。"""
    try:
        utt = build_utt(stem, names, read_textgrid(tg_path))
    except (StructError, KeyError, ValueError):
        return None
    return {s.idx: s.ref_ms for s in utt.syllables}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dialect", required=True, choices=C.DIALECTS)
    ap.add_argument("--eval-dir", required=True, action="append", help="可給多次")
    ap.add_argument("--nat-tg", required=True, help="新 MFA 對齊真人測試錄音的 TextGrid 資料夾")
    ap.add_argument("--synth-tg", default="mfa_tg_x3", help="合成語音的新 TextGrid(eval-dir 底下的相對路徑)")
    ap.add_argument("--csv-name", default="syllables_wave_x3.csv")
    args = ap.parse_args()

    nat_tgs = {p.stem: p for p in Path(args.nat_tg).expanduser().rglob("*.TextGrid")}
    if not nat_tgs:
        sys.exit(f"★ {args.nat_tg} 底下沒有 TextGrid")
    for ev in args.eval_dir:
        ev = Path(ev).expanduser()
        syn_dir = ev / args.synth_tg
        syn_tgs = {p.stem: p for p in syn_dir.rglob("*.TextGrid")}
        if not syn_tgs:
            print(f"★ {syn_dir} 沒有 TextGrid,略過")
            continue
        why, rows = Counter(), []
        old_ref, new_ref = [], []
        per = defaultdict(list)                       # (stem, idx) → 各生成種子的新預測時長
        nat_cache = {}
        for name, tgp in sorted(syn_tgs.items()):
            stem, _seed = name.rsplit("_g", 1)
            bj = ev / "bounds" / f"{stem}.json"
            if not bj.exists():
                why["沒有 bounds"] += 1
                continue
            b = json.load(open(bj, encoding="utf-8"))
            if stem not in nat_cache:
                nat_cache[stem] = nucleus_ms(stem, b["names"], nat_tgs[stem]) if stem in nat_tgs else None
            nat = nat_cache[stem]
            if nat is None:
                why["真人新對齊對不上"] += 1
                continue
            syn = nucleus_ms(stem, b["names"], tgp)
            if syn is None:
                why["合成新對齊對不上"] += 1
                continue
            for r in b["syllables"]:
                i = r["idx"]
                if i in nat and i in syn:
                    per[(stem, i)].append((syn[i], nat[i], r))
        for (stem, i), vals in sorted(per.items()):
            r = vals[0][2]
            rows.append({"dialect": args.dialect, "stem": stem, "syl_idx": i, "entering": int(r["entering"]),
                         "ref_ms": round(vals[0][1], 3), "pred_ms": round(float(np.mean([v[0] for v in vals])), 3),
                         "n_gen": len(vals)})
            old_ref.append(float(r["ref_ms"]))
            new_ref.append(vals[0][1])
        if not rows:
            print(f"★ {ev.name}:沒有任何音節對得上 {dict(why)}")
            continue
        write_rows(ev / args.csv_name, rows)
        o, n = np.array(old_ref), np.array(new_ref)
        print(f"{ev.name}:{len(rows)} 個音節;真人參考時長 新 vs 舊 平均差 {np.mean(n - o):+.1f} ms、"
              f"平均絕對差 {np.mean(np.abs(n - o)):.1f} ms、r = {pearson(o, n):.3f};略過 {dict(why) or '無'}"
              f" → {ev / args.csv_name}")


if __name__ == "__main__":
    main()
