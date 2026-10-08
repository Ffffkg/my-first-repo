"""不需要 torch / GPU 的單元測試:python -m pytest tests -q"""
import json
import pickle
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from hakka_c.analysis import eval_rows, summarize  # noqa: E402
from hakka_c.config import HOP_MS  # noqa: E402
from hakka_c.csvio import read_cols, write_rows  # noqa: E402
from hakka_c.durations import matcha_frames, merge_blank, mfa_to_frames  # noqa: E402
from hakka_c.stats import UttTable, boot_idx, delta_ci, mean_ci  # noqa: E402
from hakka_c.structure import StructError, build_utt, is_entering_syllable, load_pua_map  # noqa: E402
from hakka_c.textgrid import read_textgrid  # noqa: E402

LONG_TG = '''File type = "ooTextFile"
Object class = "TextGrid"

xmin = 0
xmax = 1.0
tiers? <exists>
size = 2
item []:
    item [1]:
        class = "IntervalTier"
        name = "words"
        xmin = 0
        xmax = 1.0
        intervals: size = 4
        intervals [1]:
            xmin = 0
            xmax = 0.1
            text = ""
        intervals [2]:
            xmin = 0.1
            xmax = 0.4
            text = "siid5"
        intervals [3]:
            xmin = 0.4
            xmax = 0.8
            text = "dong24"
        intervals [4]:
            xmin = 0.8
            xmax = 1.0
            text = ""
    item [2]:
        class = "IntervalTier"
        name = "phones"
        xmin = 0
        xmax = 1.0
        intervals: size = 8
        intervals [1]:
            xmin = 0
            xmax = 0.1
            text = ""
        intervals [2]:
            xmin = 0.1
            xmax = 0.2
            text = "s"
        intervals [3]:
            xmin = 0.2
            xmax = 0.28
            text = "ii"
        intervals [4]:
            xmin = 0.28
            xmax = 0.4
            text = "d"
        intervals [5]:
            xmin = 0.4
            xmax = 0.5
            text = "d"
        intervals [6]:
            xmin = 0.5
            xmax = 0.65
            text = "o"
        intervals [7]:
            xmin = 0.65
            xmax = 0.8
            text = "ng"
        intervals [8]:
            xmin = 0.8
            xmax = 1.0
            text = ""
'''

SHORT_TG = '''File type = "ooTextFile"
Object class = "TextGrid"

0
1.0
<exists>
1
"IntervalTier"
"phones"
0
1.0
2
0
0.4
"a"
0.4
1.0
""
'''


def test_textgrid_long_and_short(tmp_path):
    p = tmp_path / "a.TextGrid"
    p.write_text(LONG_TG, encoding="utf-8")
    t = read_textgrid(p)
    assert [x[2] for x in t["words"]] == ["", "siid5", "dong24", ""]
    assert len(t["phones"]) == 8 and t["phones"][2] == (0.2, 0.28, "ii")
    q = tmp_path / "b.TextGrid"
    q.write_text(SHORT_TG, encoding="utf-8")
    assert read_textgrid(q)["phones"] == [(0.0, 0.4, "a"), (0.4, 1.0, "")]


def test_entering_rule():
    assert is_entering_syllable("siid5") is True
    assert is_entering_syllable("bog2") is True
    assert is_entering_syllable("dong24") is False      # -ng 不是入聲
    assert is_entering_syllable("ng11") is False


def test_build_utt(tmp_path):
    p = tmp_path / "a.TextGrid"
    p.write_text(LONG_TG, encoding="utf-8")
    tiers = read_textgrid(p)
    names = ["s", "ii@5", "d", "d", "o@24", "ng"]
    u = build_utt("x", names, tiers)
    assert [s.word for s in u.syllables] == ["siid5", "dong24"]
    s0, s1 = u.syllables
    assert s0.entering and s0.nuc == 1 and s0.coda == "d" and abs(s0.ref_ms - 80) < 1e-6
    assert not s1.entering and s1.nuc == 4 and s1.coda == "ng" and s1.prepausal
    # 名稱不符要報錯,不硬湊
    with pytest.raises(StructError):
        build_utt("x", ["s", "ii@5", "b", "d", "o@24", "ng"], tiers)
    # 有停頓符號:去掉停頓後仍對得上
    u2 = build_utt("x", ["s", "ii@5", "d", "sp", "d", "o@24", "ng"], tiers)
    assert u2.syllables[0].prepausal and u2.syllables[1].nuc == 5


def test_merge_blank_rules():
    d = np.array([1, 10, 2, 20, 4], float)          # b p0 b p1 b
    assert merge_blank(d, "half").tolist() == [1 + 10 + 1, 1 + 20 + 4]
    assert merge_blank(d, "left").tolist() == [1 + 10 + 2, 20 + 4]
    assert merge_blank(d, "right").tolist() == [1 + 10, 2 + 20 + 4]
    assert merge_blank(d, "none").tolist() == [10, 20]
    for r in ("half", "left", "right"):
        assert merge_blank(d, r).sum() == d.sum()       # 守恆


def _matcha_reference(w, ls):
    """照 Matcha synthesise() + generate_path 的語意逐步模擬。"""
    w_ceil = np.ceil(w.astype(np.float32)) * np.float32(ls)       # torch 是 float32
    total = max(1, int(np.floor(w_ceil.sum(dtype=np.float32))))
    t_y = total + 8
    cum = np.cumsum(w_ceil, dtype=np.float32)
    ar = np.arange(t_y)
    masks = [(ar < c).astype(int) for c in cum]
    frames, prev = [], np.zeros(t_y, int)
    for m in masks:
        frames.append(int(((m - prev) * (ar < total)).sum()))
        prev = m
    return np.array(frames)


@pytest.mark.parametrize("ls", [1.0, 0.8, 1.13])
def test_matcha_frames_matches_matcha(ls):
    rng = np.random.default_rng(0)
    for _ in range(50):
        w = rng.gamma(2.0, 3.0, size=rng.integers(3, 40))
        assert matcha_frames(w, ls, "ceil").tolist() == _matcha_reference(w, ls).tolist()


def test_ceil_lengthens_and_cum_preserves():
    w = np.full(41, 1.2)                               # 每個符號 1.2 frame
    assert matcha_frames(w, mode="ceil").sum() == 82   # ceil → 2,總長多了 67%
    assert matcha_frames(w, mode="cum").sum() == round(41 * 1.2)


def test_mfa_to_frames():
    # 符號: s ii d sp d o ng(n=7),第 3 個是停頓
    phone_sym = [0, 1, 2, 4, 5, 6]
    iv = [(0.10, 0.20), (0.20, 0.28), (0.28, 0.40), (0.55, 0.65), (0.65, 0.80), (0.80, 0.95)]
    T = int(1.0 * 22050 / 256)
    fr = mfa_to_frames(7, phone_sym, iv, T, pause_pos=[3])
    assert len(fr) == 15 and fr.sum() == T and fr.min() >= 1
    assert fr[2 * 3 + 1] >= round(0.15 * 22050 / 256) - 1          # 0.40–0.55 的靜音給了 sp
    mb = merge_blank(fr, "half")
    want_ii = round(0.28 * 22050 / 256) - round(0.20 * 22050 / 256)
    assert mb[1] == want_ii                                            # 內部音素恰好還原
    # 有 offset:等同把所有時間往前移
    fr2 = mfa_to_frames(7, phone_sym, [(a + 0.3, b + 0.3) for a, b in iv], T, pause_pos=[3], offset_s=0.3)
    assert fr2.tolist() == fr.tolist()
    # 沒有停頓符號時,字間靜音給中間的 blank
    fr3 = mfa_to_frames(6, [0, 1, 2, 3, 4, 5], iv, T)
    assert fr3[6] > 1 and fr3.sum() == T


def test_pua_map_shapes(tmp_path):
    a, b = "", ""
    for obj in ({"sym2pua": {"s": a, "ii@5": b}, "n": 2},
                {"pua": {a: "s", b: "ii@5"}},
                {"symbols": [["s", a], ["ii@5", b]]},
                [{"symbol": "s", "char": a}, {"symbol": "ii@5", "char": b}]):
        p = tmp_path / "v.json"
        p.write_text(json.dumps(obj, ensure_ascii=False), encoding="utf-8")
        assert load_pua_map(p) == {a: "s", b: "ii@5"}


def test_stats_delta_identity():
    rng = np.random.default_rng(1)
    utt = np.repeat(np.arange(30), 5)
    et = rng.random(150) < 0.3
    e = rng.normal(5, 3, 150) + 10 * et
    t = UttTable(utt, {"e": e}, {"et": et, "non": ~et})
    idx = boot_idx(t.U, 500)
    d = delta_ci(t, "e", idx)
    assert abs(d[0] - (e[et].mean() - e[~et].mean())) < 1e-9
    assert d[1] < d[0] < d[2]
    m = mean_ci(t, "e", "et", idx)
    assert abs(m[0] - e[et].mean()) < 1e-9


def _fake_recs(n_utt=40, seed=0, bias_et=3.0):
    """假資料:真實母音入聲 ~7 frame、舒聲 ~11 frame;預測入聲多 bias_et frame。"""
    rng = np.random.default_rng(seed)
    recs = []
    for u in range(n_utt):
        n_syl = 6
        names, syls, w = [], [], [1.0]
        for k in range(n_syl):
            ent = k % 3 == 0
            ref = rng.normal(7 if ent else 11, 1.0)
            names += ["s", "a@5" if ent else "a@24", "d" if ent else "ng"]
            nuc = len(names) - 2
            w += [4.0, 1.0, ref + (bias_et if ent else 0.0) - 1.0, 1.0, 3.0, 1.0]
            syls.append({"idx": k, "word": "sad5" if ent else "sang24", "sym_idx": [nuc - 1, nuc, nuc + 1],
                         "nuc": nuc, "entering": ent, "coda": "d" if ent else "ng", "ref_ms": ref * HOP_MS,
                         "prepausal": k == n_syl - 1, "tone": "5" if ent else "24"})
        w = np.array(w)
        T = int(np.round(w.sum() * 0.95))
        recs.append({"dialect": "sixian", "stem": f"XF-{u:03d}", "spk": u % 2, "T": T, "w": w,
                     "mas": np.maximum(1, np.round(w)).astype(int), "n": len(names), "names": names,
                     "pause": [], "syllables": syls})
    return recs


def test_eval_rows_and_summary(tmp_path):
    recs = _fake_recs()
    rows = eval_rows(recs, rounding="cum")
    write_rows(tmp_path / "s.csv", rows)
    s, txt = summarize(read_cols(tmp_path / "s.csv"), n_boot=300)
    assert s["delta_bias"][0] > 2 * HOP_MS                     # 入聲多預測 3 frame
    assert s["delta_bias"][1] > 0
    s2, _ = summarize(read_cols(tmp_path / "s.csv"), n_boot=300)
    assert s2["et_vbias"] == s["et_vbias"]                     # 固定種子可重現


def test_scripts_end_to_end(tmp_path, monkeypatch):
    """c0_report / c1_calibrate / c_compare 在假資料上能從頭跑到尾。"""
    monkeypatch.setenv("HAKKA_ROOT", str(tmp_path / "hakka"))
    recs_val, recs_test = _fake_recs(30, 1), _fake_recs(40, 2)
    for nm, recs in (("val", recs_val), ("test", recs_test)):
        pickle.dump({"cond": "fake_48h", "ckpt": "x.ckpt", "protocol": "all", "recs": recs},
                    open(tmp_path / f"{nm}.pkl", "wb"))
    # c0 風格的 CSV
    from hakka_c.durations import RULES
    rows = []
    for rec in recs_test:
        src = {"mas": rec["mas"].astype(float), "cont": rec["w"], "ceil": matcha_frames(rec["w"]),
               "cum": matcha_frames(rec["w"], mode="cum")}
        for s in rec["syllables"]:
            r = {"cond": "fake_48h", "protocol": "all", "dialect": "sixian", "stem": rec["stem"], "spk": rec["spk"],
                 "gender": "F", "syl_idx": s["idx"], "word": s["word"], "tone": s["tone"],
                 "entering": int(s["entering"]), "coda": s["coda"], "prepausal": int(s["prepausal"]),
                 "ref_ms": s["ref_ms"]}
            for k, v in src.items():
                for rule in RULES:
                    r[f"{k}_{rule}_ms"] = merge_blank(v, rule)[s["nuc"]] * HOP_MS
            rows.append(r)
    write_rows(tmp_path / "c0.csv", rows)
    env = {"HAKKA_ROOT": str(tmp_path / "hakka"), "PATH": "/usr/bin:/bin"}
    py = sys.executable
    r = subprocess.run([py, str(ROOT / "c0_report.py"), str(tmp_path / "c0.csv"), "--n-boot", "200"],
                       capture_output=True, text=True, env=env)
    assert r.returncode == 0, r.stderr
    assert "恆等式檢查" in r.stdout and "偏差拆解" in r.stdout
    r = subprocess.run([py, str(ROOT / "c1_calibrate.py"), "--val", str(tmp_path / "val.pkl"),
                        "--test", str(tmp_path / "test.pkl"), "--n-boot", "200"], capture_output=True, text=True, env=env)
    assert r.returncode == 0, r.stderr
    out = tmp_path / "hakka" / "exp_c" / "c1" / "fake_48h"
    calib = json.load(open(out / "calib.json"))
    assert calib["class_cum"]["s_et"] < 1.0                    # 入聲被預測太長 → 縮放係數 < 1
    r = subprocess.run([py, str(ROOT / "c_compare.py"), "--base", str(out / "base.csv"),
                        "--method", str(out / "class_cum.csv"), "--n-boot", "200"], capture_output=True, text=True, env=env)
    assert r.returncode == 0, r.stderr
    assert "✓ 通過" in r.stdout


def test_learn_pua_map_tie(tmp_path, monkeypatch):
    """每句都恰好有一個 ii 與一個 sp(長度差都能解釋)時,要選出讓對照最一致的停頓字元。"""
    import hakka_c.pua as pua
    p = tmp_path / "a.TextGrid"
    p.write_text(LONG_TG, encoding="utf-8")
    ch = {n: chr(0xE000 + i) for i, n in enumerate(["s", "ii@5", "d", "sp", "o@24", "ng"])}
    text = "".join(ch[n] for n in ["s", "ii@5", "d", "sp", "d", "o@24", "ng"])
    monkeypatch.setattr(pua, "_sample_lines", lambda n, seed=0: [("sixian", ("/x/u%d.wav" % i, "0", text)) for i in range(20)])
    monkeypatch.setattr(pua, "_tg_index", lambda d: {"u%d" % i: p for i in range(20)})
    monkeypatch.setattr(pua, "_clean", lambda t, c=None: t)
    m, info = pua.learn_pua_map()
    assert m[ch["sp"]] == "sp" and m[ch["ii@5"]] == "ii" and m[ch["d"]] == "d"
    assert info["pause_fit"] == 1.0 and not info["low_purity"]


def test_margin_offset():
    from hakka_c.durations import margin_offset
    m = 0.05
    # 一般情況:前後都裁到 50 ms
    assert abs(margin_offset(0.60, 2.00, 2.50, 2.00 + m - (0.60 - m), m) - 0.55) < 1e-9
    # 原檔第一個音素前不到 50 ms:起點 0
    assert margin_offset(0.03, 2.00, 2.50, 2.05, m) == 0.0
    # 原檔最後一個音素後不到 50 ms:句尾被原檔長度截斷,仍可接受
    assert abs(margin_offset(0.60, 2.48, 2.50, 2.50 - 0.55, m) - 0.55) < 1e-9
    # 沒有被裁切(長度等於原檔)
    assert margin_offset(0.60, 2.00, 2.50, 2.50, m) == 0.0
    # 對不上任何情況
    assert margin_offset(0.60, 2.00, 2.50, 1.20, m) is None
