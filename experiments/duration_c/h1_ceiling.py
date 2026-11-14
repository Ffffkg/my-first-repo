"""H1:時長 ρ 的「上限」在哪裡?先量清楚,再決定要不要投入更重的訓練方法(例如自監督預訓練)。不需要 GPU。

三種診斷,都在同一批測試句、同樣的入聲母音核上算 ρ 與 ET-VDE(與 c_compare / c_seeds 相同的定義):

1. 語速天花板(oracle rate):真人每一句的整體語速是文字裡看不到的(同一句可以念快或念慢)。
   把每一句的預測整體乘上一個係數,讓該句「全部母音核」的平均 log 比值 = 0,再算 ρ。
   這是作弊的上限(要先知道真人這一句念多快),只用來看「剩下的誤差有多少是整句快慢造成的」。
2. 純文字特徵模型(GBM):不用聲學模型,只用文字能得到的特徵(調、韻尾、母音、聲母、前後音節、
   句中位置、離停頓多遠、語者)訓練梯度提升樹,直接預測真人母音核時長。
     * 用 1h 訓練 → 跟 C6 同樣資料量,看「顯式語言特徵」本身能到多少
     * 用大量訓練資料(train_full,抽樣)→ 文字可預測部分的大致上限
3. 各系統(base / C2 / C6 …,多種子)的原始 ρ 與 oracle-rate ρ 並列。

    python h1_ceiling.py --dialect sixian \\
        --sys base=$E/base_sixian_step40000,$E/base_sixian_s123_step40000,$E/base_sixian_s2026_step40000 \\
        --sys c6=$E/c6_sixian_step40000_cum,...
"""
import argparse
import random
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from hakka_c import config as C  # noqa: E402
from hakka_c.csvio import boolcol, read_cols  # noqa: E402
from hakka_c.stats import pearson  # noqa: E402
from hakka_c.structure import StructError, base_of, build_utt  # noqa: E402
from hakka_c.textgrid import read_textgrid  # noqa: E402

CAT = ["tone", "coda", "nuc", "onset", "spk", "p_tone", "p_coda", "n_tone", "n_coda", "n_onset"]
NUM = ["entering", "prepausal", "final", "idx", "from_end", "n_syl", "to_pause", "from_pause", "p_prepausal",
       "n_entering", "p_entering"]


def utt_rows(utt, names, stem, spk):
    syls = utt.syllables
    n = len(syls)
    pp = [i for i, s in enumerate(syls) if s.prepausal]
    rows = []
    for i, s in enumerate(syls):
        onset = base_of(names[s.sym_idx[0]]) if s.sym_idx and s.sym_idx[0] != s.nuc else "-"
        p, nx = (syls[i - 1] if i else None), (syls[i + 1] if i + 1 < n else None)
        nxt_pause = min([j for j in pp if j >= i], default=n - 1)
        prv_pause = max([j for j in pp if j < i], default=-1)
        nx_onset = base_of(names[nx.sym_idx[0]]) if nx and nx.sym_idx and nx.sym_idx[0] != nx.nuc else "-"
        rows.append({
            "stem": stem, "syl_idx": s.idx, "ref_ms": s.ref_ms, "spk": spk,
            "tone": s.tone, "coda": s.coda or "-", "nuc": base_of(names[s.nuc]), "onset": onset,
            "entering": int(s.entering), "prepausal": int(s.prepausal), "final": int(i == n - 1),
            "idx": i, "from_end": n - 1 - i, "n_syl": n, "to_pause": nxt_pause - i, "from_pause": i - prv_pause - 1,
            "p_tone": p.tone if p else "#", "p_coda": (p.coda or "-") if p else "#",
            "p_prepausal": int(p.prepausal) if p else 1, "p_entering": int(p.entering) if p else 0,
            "n_tone": nx.tone if nx else "#", "n_coda": (nx.coda or "-") if nx else "#",
            "n_onset": nx_onset, "n_entering": int(nx.entering) if nx else 0,
        })
    return rows


def extract(dialect, split, max_utts, cleaners="hakka_cleaners"):
    sys.path.insert(0, str(C.MATCHA))
    from matcha.text import text_to_sequence
    from hakka_c.matcha_io import read_filelist, textgrid_index
    from hakka_c.pua import resolve_pua_map

    pua, _ = resolve_pua_map((cleaners,), verbose=False)
    rows_in = read_filelist(C.filelist("joint", dialect, split))
    if max_utts and len(rows_in) > max_utts:
        rows_in = random.Random(C.SPLIT_SEED).sample(rows_in, max_utts)
    tg_idx = textgrid_index(dialect)
    out, why = [], Counter()
    for r in rows_in:
        stem = Path(r["wav"]).stem
        try:
            _, clean = text_to_sequence(r["text"], [cleaners])
            names = [pua[c] for c in clean]
            if stem not in tg_idx:
                raise StructError("找不到 TextGrid")
            utt = build_utt(stem, names, read_textgrid(tg_idx[stem]))
        except (StructError, KeyError, ValueError) as e:
            why[str(e)[:40]] += 1
            continue
        out += utt_rows(utt, names, stem, str(r["spk"]))
    print(f"  {split}:{len({r['stem'] for r in out})} 句、{len(out)} 個音節(跳過 {sum(why.values())})")
    return out


class Encoder:
    def __init__(self, rows):
        self.maps = {k: {v: i for i, v in enumerate(sorted({str(r[k]) for r in rows}))} for k in CAT}

    def __call__(self, rows):
        X = np.array([[self.maps[k].get(str(r[k]), -1) for k in CAT] + [float(r[k]) for k in NUM] for r in rows], float)
        X[X == -1] = np.nan                   # 沒看過的類別 → 缺值(HistGBM 會處理)
        return X


def fit_gbm(train_rows, seed=0):
    from sklearn.ensemble import HistGradientBoostingRegressor
    enc = Encoder(train_rows)
    cat_mask = [len(enc.maps[k]) <= 250 for k in CAT] + [False] * len(NUM)
    X = enc(train_rows)
    y = np.log([r["ref_ms"] for r in train_rows])
    m = HistGradientBoostingRegressor(max_iter=600, learning_rate=0.05, max_leaf_nodes=31, min_samples_leaf=20,
                                      l2_regularization=1.0, categorical_features=cat_mask,
                                      early_stopping=True, validation_fraction=0.1, random_state=seed)
    m.fit(X, y)
    return lambda rows: np.exp(m.predict(enc(rows)))


def scores(stem, et, ref, pred):
    """原始與 oracle-rate 的 ρ、ET-VDE(只看入聲母音核)。"""
    pred = np.asarray(pred, float)
    lr = np.log(ref) - np.log(np.maximum(pred, 1e-3))
    adj = pred.copy()
    for s in set(stem):
        m = stem == s
        adj[m] = pred[m] * np.exp(lr[m].mean())
    return (pearson(pred[et], ref[et]), np.abs(pred[et] - ref[et]).mean(),
            pearson(adj[et], ref[et]), np.abs(adj[et] - ref[et]).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dialect", required=True, choices=C.DIALECTS)
    ap.add_argument("--sys", action="append", default=[], metavar="NAME=DIR1,DIR2,...", help="d1 評估資料夾(多種子)")
    ap.add_argument("--small", default="ladder_1h", help="小訓練集(與 C6 相同)")
    ap.add_argument("--big", default="train_full", help="大訓練集")
    ap.add_argument("--big-max", type=int, default=8000)
    ap.add_argument("--out")
    args = ap.parse_args()

    systems = []
    for spec in args.sys:
        name, dirs = spec.split("=", 1)
        systems.append((name, [read_cols(Path(x).expanduser() / "syllables_wave.csv") for x in dirs.split(",") if x]))
    if not systems:
        sys.exit("★ 至少給一個 --sys(用來決定測試句與音節)")
    ref_csv = systems[0][1][0]
    keep = {(s, int(i)) for s, i in zip(ref_csv["stem"], ref_csv["syl_idx"])}

    print("抽文字特徵 …")
    test = [r for r in extract(args.dialect, "test", 0) if (r["stem"], int(r["syl_idx"])) in keep]
    small = extract(args.dialect, args.small, 0)
    big = extract(args.dialect, args.big, args.big_max)
    stems_test = {r["stem"] for r in test}
    small = [r for r in small if r["stem"] not in stems_test]
    big = [r for r in big if r["stem"] not in stems_test]

    stem = np.array([r["stem"] for r in test])
    et = np.array([bool(r["entering"]) for r in test])
    ref = np.array([r["ref_ms"] for r in test], float)
    L = [f"# H1 時長 ρ 的上限診斷:{args.dialect}\n",
         f"測試句 {len(stems_test)} 句、入聲母音核 {et.sum()} 個。oracle-rate = 每句整體快慢用真人校正後(作弊上限)。\n",
         "| 系統 | ρ | ET-VDE (ms) | ρ(oracle-rate) | ET-VDE(oracle-rate) |", "|---|---|---|---|---|"]
    for name, rows in ((f"文字特徵 GBM・{args.small}", small), (f"文字特徵 GBM・{args.big}(≤{args.big_max} 句)", big)):
        print(f"訓練 {name}({len(rows)} 個音節)…")
        pred = fit_gbm(rows)(test)
        r = scores(stem, et, ref, pred)
        L.append(f"| {name} | {r[0]:.3f} | {r[1]:.1f} | {r[2]:.3f} | {r[3]:.1f} |")
    for name, seeds in systems:
        rs = []
        for c in seeds:
            e = boolcol(c["entering"])
            rs.append(scores(np.asarray(c["stem"]), e, c["ref_ms"], c["pred_ms"]))
        rs = np.array(rs)
        mean = rs.mean(axis=0)
        L.append(f"| {name}({len(seeds)} 種子平均) | {mean[0]:.3f} | {mean[1]:.1f} | {mean[2]:.3f} | {mean[3]:.1f} |")
    L += ["", "讀法:",
          "* 系統的 ρ(oracle-rate) 若明顯高於 ρ:有一大塊誤差來自「整句念快或念慢」,這是文字看不到的,任何訓練方法都很難消除。",
          "* GBM・大資料 的 ρ 約等於「只靠文字能預測到的上限」;C6 若已接近它,再加模型容量或自監督也不容易再往上。",
          "* GBM・1h 若高於 C6:顯式語言特徵(調、韻尾、離停頓距離)還有沒被時長預測器用到的資訊,值得加進模型。"]
    text = "\n".join(L)
    print("\n" + text)
    out = Path(args.out).expanduser() if args.out else C.OUT_ROOT / "h1" / f"ceiling_{args.dialect}.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text, encoding="utf-8")
    print(f"\n→ {out}")


if __name__ == "__main__":
    main()
