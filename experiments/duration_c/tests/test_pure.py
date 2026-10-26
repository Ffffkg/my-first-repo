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


def test_merge_blank_halfcap():
    d = np.array([1, 10, 1, 20, 9], float)          # b p0 b p1 b;最後一個 blank 裡有 8 frame 的靜音
    assert merge_blank(d, "halfcap").tolist() == [1 + 10 + 0.5, 0.5 + 20 + 1]
    assert merge_blank(d, "half").tolist() == [1 + 10 + 0.5, 0.5 + 20 + 9]
    # 訓練目標的借 frame 規則:halfcap 把借走的還回去 → 內部音素恰好是 MFA 的 frame 數,靜音不算進去
    phone_sym = [0, 1, 2]
    iv = [(0.10, 0.20), (0.20, 0.30), (0.30, 0.45)]
    fr = mfa_to_frames(3, phone_sym, iv, int(0.6 * 22050 / 256))
    mfa = [round(b * 22050 / 256) - round(a * 22050 / 256) for a, b in iv]
    assert merge_blank(fr, "halfcap")[1] == mfa[1]
    assert merge_blank(fr, "none")[1] == mfa[1] - 1                      # 只看符號本身會少 1 frame


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


def test_c3_components():
    sys.path.insert(0, str(ROOT))
    from c3_target_check import syllable_components
    d = np.array([2, 3, 1, 5, 4])                        # blank, 子音, blank, 母音, 句尾 blank(含 3 frame 靜音)
    syl = [{"nuc": 1, "coda": "", "ref_ms": 9.5 * HOP_MS}]
    c = syllable_components(d, d.astype(float), syl)[0]
    assert c["A"] == pytest.approx(0, abs=1e-3)          # half:5 + 0.5·1 + 1·4 = 9.5 frame
    assert c["S"] == pytest.approx(3 * HOP_MS, abs=1e-3)
    assert c["B"] == pytest.approx(0, abs=1e-3) and c["R"] == pytest.approx(0, abs=1e-3) and c["N"] == 0
    c = syllable_components(d, d - 0.3, syl)[0]           # 預測略短 → ceil 補回同樣的整數
    assert c["B"] == pytest.approx(-0.75 * HOP_MS, abs=1e-3)
    assert c["R"] == pytest.approx(0.75 * HOP_MS, abs=1e-3)
    assert c["total"] == pytest.approx(c["A"] + c["B"] + c["R"], abs=1e-2)
    assert c["open"] == 1 and c["blank_r_tgt"] == 4


def test_s_pp_scales_only_prepausal_nucleus():
    from hakka_c.analysis import scaled_w
    rec = {"w": np.full(9, 4.0), "syllables": [
        {"nuc": 1, "entering": False, "prepausal": False}, {"nuc": 3, "entering": True, "prepausal": True}]}
    w = scaled_w(rec, s_et=0.5, s_pp=0.5)
    assert w[3] == 4.0 and w[7] == 1.0 and w[[0, 1, 2, 4, 5, 6, 8]].tolist() == [4.0] * 7


def test_fit_s_pp_recovers_overlength():
    sys.path.insert(0, str(ROOT))
    from c3_fit_pp import fit_s_pp, nucleus_errors
    recs = []
    for k in range(20):                                   # 停頓前母音核目標 8 frame,預測 10 → s_pp ≈ 0.8
        d = np.array([1, 3, 1, 6, 1, 3, 1, 8, 1])
        w = d.astype(float) - 0.5
        w[7] = 9.5
        recs.append({"d": d, "w": w, "syllables": [
            {"nuc": 1, "entering": False, "prepausal": False}, {"nuc": 3, "entering": False, "prepausal": True}]})
    s = fit_s_pp(recs)
    assert 0.75 <= s <= 0.84
    e, _, pp = nucleus_errors(recs, s)
    assert abs(e[pp].mean()) < 1e-6 and abs(e[~pp].mean()) < 1e-6


def test_pick_rounding_follows_predictor_bias():
    sys.path.insert(0, str(ROOT))
    from c6_pick_rounding import pick

    def recs(offset):                                     # w = 目標 + offset:offset≈0 → ceil 偏長 → 選 cum
        out = []
        for k in range(10):
            d = np.array([1, 4, 1, 6, 1, 5, 1, 7, 1])
            w = d + offset + 0.1 * np.sin(np.arange(9) + k)
            out.append({"d": d, "w": w, "syllables": [
                {"nuc": 1, "entering": True, "prepausal": False}, {"nuc": 3, "entering": False, "prepausal": False}]})
        return out
    assert pick(recs(0.0))[0] == "cum"
    assert pick(recs(-0.5))[0] == "ceil"                  # 預測器偏短半格:ceil 剛好補回


def test_geo_mean_w_and_probe_ensemble_gain(tmp_path):
    """多個種子的連續時長取幾何平均:獨立雜訊被平均掉,ρ 變高、誤差變小。"""
    from hakka_c.analysis import geo_mean_w
    assert np.allclose(geo_mean_w([[1.0, 4.0], [4.0, 1.0]]), [2.0, 2.0])
    sys.path.insert(0, str(ROOT))
    import h3_probe as P
    truth = _fake_recs(60, 5, bias_et=0.0)
    dirs = []
    for k in range(3):
        rng = np.random.default_rng(100 + k)
        recs = []
        for rec in truth:
            w = np.array(rec["w"], float)
            for s in rec["syllables"]:
                w[2 * s["nuc"] + 1] *= float(np.exp(rng.normal(0, 0.25)))      # 各種子獨立的預測雜訊
            recs.append(dict(rec, w=w))
        d = tmp_path / f"seed{k}"
        d.mkdir()
        pickle.dump({"params": {"rounding": "cum"}, "recs": recs}, open(d / "recs.pkl", "wb"))
        dirs.append(str(d))
    stem, idx, et, ref, Pm, ens, rds = P.system_tables(dirs)
    single = P.mean_metrics(et, ref, Pm)
    ensm = P.metrics(et, ref, ens)
    assert ensm["et_vde"] < single["et_vde"]
    assert ensm["et_rho"] > single["et_rho"]
    ci = P.boot_ci(stem, et, ref, Pm, ens, ["et_vde"], n_boot=200)
    assert ci["et_vde"][2] < 0                                        # 誤差下降的 CI 上界 < 0


def test_h3_xdialect_blend_runs(tmp_path, monkeypatch):
    """跨腔 GBM + 神經混合在假資料上能跑完,且來源腔資料在目標腔很少時有幫助。"""
    sys.path.insert(0, str(ROOT))
    import h1_ceiling as H
    import h3_xdialect as X
    monkeypatch.setenv("HAKKA_ROOT", str(tmp_path / "hakka"))
    rng = np.random.default_rng(0)

    def mk(tag, n_utt):
        rows = []
        for u in range(n_utt):
            for i in range(8):
                coda = str(rng.choice(["-", "d", "g", "n", "ng"]))
                nuc = str(rng.choice(["a", "i", "u", "e"]))
                et = coda in ("d", "g")
                pp = int(i == 7)
                ref = 60 + (25 if not et else 0) + (15 if nuc == "a" else 0) + 30 * pp + rng.normal(0, 8)
                rows.append({"stem": f"{tag}{u}", "syl_idx": i, "ref_ms": max(ref, 20), "spk": "0",
                             "tone": str(rng.choice(["1", "2", "3"])), "coda": coda, "nuc": nuc, "onset": "s",
                             "entering": int(et), "prepausal": pp, "final": pp, "idx": i, "from_end": 7 - i, "n_syl": 8,
                             "to_pause": 7 - i, "from_pause": i, "p_tone": "#", "p_coda": "#", "p_prepausal": 0,
                             "p_entering": 0, "n_tone": "#", "n_coda": "#", "n_onset": "-", "n_entering": 0})
        return rows
    cache = {}

    def fake_extract(dialect, split, max_utts, cleaners="hakka_cleaners"):
        n = {"test": 60, "val": 40, "ladder_1h": 6, "ladder_10h": 100, "ladder_48h": 400}[split]
        cache.setdefault((dialect, split), mk(split[:3] + dialect[:1], n))
        return [dict(r) for r in cache[(dialect, split)]]
    monkeypatch.setattr(H, "extract", fake_extract)

    def nn_recs(rows, seed):
        r = np.random.default_rng(seed)
        by = {}
        for x in rows:
            by.setdefault(x["stem"], []).append(x)
        recs = []
        for stem, rs in by.items():
            w, syl = [1.0], []
            for j, x in enumerate(rs):
                w += [3.0, 1.0, x["ref_ms"] / 11.61 * float(np.exp(r.normal(0, 0.3))), 1.0]
                syl.append({"idx": x["syl_idx"], "nuc": 2 * j + 1, "entering": bool(x["entering"]), "ref_ms": x["ref_ms"]})
            recs.append({"stem": stem, "w": np.array(w), "syllables": syl})
        return recs
    dirs, vals = [], []
    for k in range(2):
        d = tmp_path / f"t{k}"
        d.mkdir()
        pickle.dump({"recs": nn_recs(fake_extract("sixian", "test", 0), k)}, open(d / "recs.pkl", "wb"))
        pickle.dump({"recs": nn_recs(fake_extract("sixian", "val", 0), 10 + k)}, open(tmp_path / f"v{k}.pkl", "wb"))
        dirs.append(str(d))
        vals.append(str(tmp_path / f"v{k}.pkl"))
    out = tmp_path / "x.md"
    monkeypatch.setattr(sys, "argv", ["h3_xdialect.py", "--dialect", "sixian", "--source", "hailu", "--valid-list", "val",
                                      "--c6val", ",".join(vals), "--c6test", ",".join(dirs), "--out", str(out)])
    X.main()
    text = out.read_text(encoding="utf-8")
    assert "零樣本" in text and "集成 + GBM" in text
    rho = {}
    for line in text.splitlines():
        if line.startswith("| 文字特徵 GBM・目標腔 ladder_1h"):
            rho["t1"] = float(line.split("|")[2])
        if "來源 + 目標 ladder_1h(目標權重 10)" in line:
            rho["st"] = float(line.split("|")[2])
    assert rho["st"] > rho["t1"]


def test_f0_nucleus_points_on_glide(tmp_path):
    from hakka_c.f0feat import f0_track, nucleus_points
    sr = 22050
    t = np.arange(int(sr * 1.0)) / sr
    f = np.where(t < 0.5, 150.0, 150.0 + 100.0 * (t - 0.5) / 0.5)      # 前半平、後半上升到 250 Hz
    x = 0.3 * np.sin(2 * np.pi * np.cumsum(f) / sr)
    f0, tt = f0_track(x, sr)
    flat, vf = nucleus_points(f0, tt, 0.1, 0.4)
    rise, _ = nucleus_points(f0, tt, 0.55, 0.95)
    assert vf > 0.9 and all(abs(2 ** p - 150) < 5 for p in flat)
    assert rise[0] < rise[-1] and abs(2 ** rise[-1] - 240) < 10


def test_g1_synth_features(tmp_path):
    """假的 eval 資料夾:bounds + 合成 TextGrid + 音檔,檢查母音核 F0 取對位置。"""
    import json as _json
    import soundfile as sf
    from dataclasses import asdict
    ev = tmp_path / "ev"
    for d in ("wav", "bounds", "mfa_tg/XF"):
        (ev / d).mkdir(parents=True)
    (ev / "mfa_tg/XF/XF-1_g1234.TextGrid").write_text(LONG_TG, encoding="utf-8")
    names = ["s", "ii@5", "d", "d", "o@24", "ng"]
    tg = tmp_path / "a.TextGrid"
    tg.write_text(LONG_TG, encoding="utf-8")
    u = build_utt("XF-1", names, read_textgrid(tg))
    _json.dump({"names": names, "syllables": [asdict(s) for s in u.syllables]},
               open(ev / "bounds/XF-1.json", "w"), ensure_ascii=False)
    sr = 22050
    t = np.arange(sr) / sr
    f = np.where(t < 0.45, 120.0, 220.0)                               # ii(0.20–0.28)=120 Hz、o(0.50–0.65)=220 Hz
    sf.write(ev / "wav/XF-1_g1234.wav", 0.3 * np.sin(2 * np.pi * np.cumsum(f) / sr), sr)
    out = tmp_path / "f.csv"
    subprocess.run([sys.executable, str(ROOT / "g1_tone_feats.py"), "synth", "--dialect", "sixian",
                    "--eval-dir", str(ev), "--out", str(out)], check=True, capture_output=True)
    c = read_cols(out)
    assert c["word"].tolist() == ["siid5", "dong24"] and c["gen_seed"].tolist() == [1234, 1234]
    assert abs(2 ** c["p3"][0] - 120) < 6 and abs(2 ** c["p3"][1] - 220) < 8
    assert abs(c["dur_ms"][0] - 80) < 0.1 and c["entering"].tolist() == [1, 0]


def _words_tg(words):
    n = len(words) + 2
    L = ['File type = "ooTextFile"', 'Object class = "TextGrid"', "", "xmin = 0", f"xmax = {n * 0.1:.1f}", "tiers? <exists>",
         "size = 1", "item []:", "    item [1]:", '        class = "IntervalTier"', '        name = "words"', "        xmin = 0",
         f"        xmax = {n * 0.1:.1f}", f"        intervals: size = {n}"]
    for i, w in enumerate([""] + words + [""]):
        L += [f"        intervals [{i + 1}]:", f"            xmin = {i * 0.1:.1f}", f"            xmax = {(i + 1) * 0.1:.1f}",
              f'            text = "{w}"']
    return "\n".join(L) + "\n"


def test_a1_text_utils():
    sys.path.insert(0, str(ROOT))
    import a1_asr_check as A
    assert A.letters("Sâd5") == "sad" and A.letters("ngiàn2") == "ngian"
    assert A.with_marks("sàd") != A.with_marks("sad") and A.with_marks("sad 5!") == "sad5"
    assert A.tokens("Sad5, ngin-ngin!") == ["sad5", "ngin", "ngin"]
    assert A.lev("kitten", "sitting") == 3
    assert A.align_tokens(["sad", "ngin", "kam"], ["sad", "kam"]) == [1.0, 0.0, 1.0]          # 被刪掉的 token = 0
    assert A.align_tokens(["sad"], ["sat"])[0] == pytest.approx(2 / 3)


REAL_PFS = "sàt ngîn kuk kám hó"            # 假的真人辨識結果(白話字風格):入聲字 sàt、kuk,其他字 ngîn、kám、hó


def _a1_fake_rows(n=60):
    """每句 5 個字。ok = 與真人相同;bad = 入聲字掉了韻尾;tone = 聲調記號全錯(字母不變)。回傳 (標準答案音節數, 真人, 合成)。"""
    ref_n = {f"s{i}": 5 for i in range(n)}
    real = {st: REAL_PFS for st in ref_n}
    syn = {}
    for st in ref_n:
        syn[("ok", "0", st)] = REAL_PFS
        syn[("bad", "0", st)] = "sà ngîn ku kám hó"
        syn[("tone", "0", st)] = "sât ngìn kuk kam hô"
    return ref_n, real, syn


def test_a1_tokens_ignore_hallucinations_and_read_tone_marks():
    sys.path.insert(0, str(ROOT))
    import a1_asr_check as A
    assert A.latin_tokens("2004년 300 piong-to 切 ⁇ chhut-loi") == [("piong", ""), ("to", ""), ("chhut", ""), ("loi", "")]
    assert A.latin_tokens("pûn-ngìn kit-tó sṳ̍t `m") == [("pun", "\u0302"), ("ngin", "\u0300"), ("kit", ""), ("to", "\u0301"),
                                                    ("sut", "\u030d"), ("m", "")]                 # ṳ 的下加點不算聲調記號
    assert A.ends_stop("kit") and A.ends_stop("chhok") and A.ends_stop("pap") and not A.ends_stop("ngin") and not A.ends_stop("khi")


def _section(text, start, end=None):
    a = text.index(start)
    return text[a:text.index(end, a) if end else len(text)]


def test_a1_metrics_flag_degraded_entering_words():
    sys.path.insert(0, str(ROOT))
    import a1_asr_check as A
    ref_n, real, syn = _a1_fake_rows()
    M = A.per_stem_metrics(real, ref_n, syn)
    tot = lambda s, k: sum(M[s][k].values())                                                       # noqa: E731
    assert tot("ok", "n_et") == 2 * 60 and tot("ok", "n_non") == 3 * 60
    assert tot("ok", "len_letters") == 15 * 60 and tot("ok", "ed_letters") == 0 and tot("bad", "ed_letters") == 2 * 60
    assert tot("ok", "kept_et") == tot("ok", "n_et") and tot("ok", "tone_et") + tot("ok", "tone_non") == 5 * 60
    # bad:入聲字 sàt→sà、kuk→ku 相似度 2/3,韻尾沒保留;其他字完全相同
    assert tot("bad", "sim_et") / tot("bad", "n_et") == pytest.approx(2 / 3) and tot("bad", "kept_et") == 0
    assert tot("bad", "sim_non") == tot("bad", "n_non") and tot("bad", "false_non") == 0
    assert tot("bad", "tone_et") == tot("bad", "m_et") and tot("bad", "tone_non") == tot("bad", "m_non")      # 聲調記號沒變
    # tone:字母不變、聲調記號全錯(入聲字 sàt→sât 錯,kuk 無記號仍對;其他字全錯)
    assert tot("tone", "sim_et") == tot("tone", "n_et") and tot("tone", "kept_et") == tot("tone", "n_et")
    assert tot("tone", "tone_non") == 0 and tot("tone", "tone_et") == tot("tone", "m_et") / 2
    text = A.build_report(M, ["ok", "bad", "tone"], "ok", n_boot=200)
    assert "120 個入聲字、180 個其他字" in text and "## 摘要" in text
    summary = _section(text, "## 摘要", "## 1.")
    assert "| bad |" in summary and "| tone |" in summary and "| ok |" not in summary               # base 不放進摘要
    sec2b = _section(text, "**2b.", "## 3.")
    assert [ln for ln in sec2b.splitlines() if ln.startswith("| ok |")][0].startswith("| ok | 100.0 [100.0, 100.0] |")   # ok 的入聲保留率 100%
    assert [ln for ln in sec2b.splitlines() if ln.startswith("| bad |")][0].startswith("| bad | 0.0 [0.0, 0.0] |")       # bad 的入聲保留率 0%
    sec2a = _section(text, "**2a.", "**2b.")
    bad_row = [ln for ln in sec2a.splitlines() if ln.startswith("| bad |")][0]
    assert bad_row.count("-33.3 [-33.3, -33.3]") == 3            # 入聲字相似度差、入聲−其他差距、差距與 ok 的差(ok 的差距為 0)
    # 入聲字太少 → 第 2 節略過、摘要不放入聲欄位,其餘照常;完全沒有資料 → 友善訊息
    small = {st: v for st, v in list(real.items())[:5]}
    M3 = A.per_stem_metrics(small, ref_n, {k: v for k, v in syn.items() if k[2] in small})
    text3 = A.build_report(M3, ["ok", "bad"], "ok", n_boot=200)
    assert "入聲字太少" in text3 and "## 1." in text3 and "## 3." in text3 and "入聲保留率" not in _section(text3, "## 摘要", "## 1.")
    assert "沒有任何可用" in A.build_report({}, ["ok"], "ok")


def test_a1_unreliable_real_transcripts_are_excluded():
    """整句被辨識成泰文 / 漢字數字的真人錄音,不能當參考答案(否則 CER 的分母趨近 0 而爆表),整句排除。"""
    sys.path.insert(0, str(ROOT))
    import a1_asr_check as A
    ref_n, real, syn = _a1_fake_rows(10)
    real["s0"] = "สวัสดี ครับ"
    real["s1"] = "2004년 300 切 sàt"
    M = A.per_stem_metrics(real, ref_n, syn)
    assert set(M["ok"]["ed_letters"]) == {f"s{i}" for i in range(2, 10)}
    assert A.reliable(REAL_PFS, 5) and not A.reliable("สวัสดี ครับ", 5) and not A.reliable("sat", 0) and A.reliable("sat ngin", 0)
    assert A.is_odd_char("ส") and A.is_odd_char("년") and A.is_odd_char("3") and not A.is_odd_char("ṳ") and not A.is_odd_char("a")


def test_a1_to16k_and_reference_syllables(tmp_path, monkeypatch):
    sys.path.insert(0, str(ROOT))
    import a1_asr_check as A
    import soundfile as sf
    from hakka_c import config as C
    sr = 22050
    sf.write(tmp_path / "a.wav", 0.3 * np.sin(2 * np.pi * 220 * np.arange(sr) / sr), sr)
    A.to16k(tmp_path / "a.wav", tmp_path / "out" / "a16.wav")
    y, sr2 = sf.read(tmp_path / "out" / "a16.wav")
    assert sr2 == 16000 and abs(len(y) - 16000) <= 2
    monkeypatch.setattr(C, "HAKKA", tmp_path / "hakka")
    d = C.textgrid_dir("sixian") / "XF"
    d.mkdir(parents=True)
    (d / "XF-1.TextGrid").write_text(_words_tg(["sad5", "ngin2", "sp", "gug5"]), encoding="utf-8")
    assert A.reference_syllables("sixian", ["XF-1", "nope"]) == {"XF-1": ["sad5", "ngin2", "gug5"]}     # 靜音標籤不算音節


def _a1_fixture(tmp_path, monkeypatch, dialects=("sixian",), n=30, gens=(1234,)):
    """假的資料夾:TextGrid(標準答案)、測試清單、真人錄音、三個系統(ok / bad / new)的合成音檔;
    回傳 (a1_asr_check 模組, 辨識次數紀錄, run(腔調, 系統…, extra=…))。ASR 用假的:bad* 資料夾的音檔入聲字掉韻尾。"""
    sys.path.insert(0, str(ROOT))
    import a1_asr_check as A
    from hakka_c import config as C
    monkeypatch.setattr(C, "HAKKA", tmp_path / "hakka")
    monkeypatch.setattr(C, "MATCHA", tmp_path / "matcha")
    for dia in dialects:
        pre = "XF" if dia == "sixian" else "HF"
        stems = [f"{pre}-{i}" for i in range(n)]
        tg = C.textgrid_dir(dia) / pre
        tg.mkdir(parents=True)
        fl = C.filelist("joint", dia, "test")
        fl.parent.mkdir(parents=True)
        (tmp_path / dia / "real").mkdir(parents=True)
        for st in stems:
            (tg / f"{st}.TextGrid").write_text(_words_tg(["sad5", "ngin2", "gug5", "kam24", "ho31"]), encoding="utf-8")
            (tmp_path / dia / "real" / f"{st}.wav").write_bytes(b"x")
        fl.write_text("".join(f"{tmp_path / dia / 'real' / (st + '.wav')}|0|text\n" for st in stems), encoding="utf-8")
        for name in ("ok", "bad", "new"):
            (tmp_path / dia / name / "wav").mkdir(parents=True)
            for st in stems:
                for gs in gens:
                    (tmp_path / dia / name / "wav" / f"{st}_g{gs}.wav").write_bytes(b"x")

    def fake_to16k(src, dst):
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_bytes(b"x")
    monkeypatch.setattr(A, "to16k", fake_to16k)
    calls = []

    def fake_omni(files, model, lang, batch, chunk=64):
        calls.append(len(files))
        assert lang == "hak_Latn" and model == "fake-model"
        return ["sà ngîn ku kám hó" if f.parent.name.startswith("bad") else REAL_PFS for f in files]
    monkeypatch.setattr(A, "transcribe_omni", fake_omni)

    def run(dia, *names, extra=()):
        argv = ["a1", "run", "--dialect", dia, "--out-dir", str(tmp_path / "out" / dia), "--base", "ok", "--model", "fake-model",
                "--n-boot", "100", *extra]
        for n in names:
            argv += ["--sys", f"{n}={tmp_path / dia / n}"]
        monkeypatch.setattr(sys, "argv", argv)
        A.main()
    return A, calls, run


def test_a1_run_end_to_end_with_fake_asr(tmp_path, monkeypatch):
    """只檢查檔案流程(找音檔、16k 快取、可續跑、只辨識新增的系統)與報告產出。"""
    A, calls, run = _a1_fixture(tmp_path, monkeypatch)
    run("sixian", "ok", "bad")
    assert calls == [90]                                              # 30 句真人 + 2 個系統 × 30 句
    rep = (tmp_path / "out" / "sixian" / "a1_omni_fake-model_hak_Latn_sixian.md").read_text(encoding="utf-8")
    assert "60 個入聲字、90 個其他字" in rep and "| bad |" in rep and "辨識輸出範例" in rep and "## 摘要" in rep
    run("sixian", "ok", "bad")
    assert calls == [90]                                              # 全部做過了:不再呼叫 ASR
    run("sixian", "ok", "bad", "new")
    assert calls == [90, 30]                                          # 只辨識新增的系統
    csv_path = tmp_path / "out" / "sixian" / "transcripts_omni_fake-model_hak_Latn.csv"
    assert len(A.read_tsv_rows(csv_path)) == 30 * 4
    monkeypatch.setattr(sys, "argv", ["a1", "analyze", "--dialect", "sixian", "--out-dir", str(tmp_path / "out" / "sixian"),
                                      "--base", "ok", "--n-boot", "100", "--model", "fake-model"])
    A.main()                                                          # analyze 只重算報告,不碰 ASR
    assert calls == [90, 30]
    monkeypatch.setattr(sys, "argv", ["a1", "analyze", "--dialect", "sixian", "--out-dir", str(tmp_path / "out" / "sixian"),
                                      "--model", "other"])
    with pytest.raises(SystemExit):                                   # 換了模型 = 另一個辨識結果檔,不會讀到舊的
        A.main()


def test_a1_pool_two_dialects_and_extra_gen_seeds(tmp_path, monkeypatch):
    A, calls, run = _a1_fixture(tmp_path, monkeypatch, dialects=("sixian", "hailu"), gens=(1234, 2345))
    run("sixian", "ok", "bad")
    run("hailu", "ok", "bad")
    assert calls == [90, 90]                                          # 每腔:30 句真人 + 2 系統 × 30 句
    monkeypatch.setattr(sys, "argv", ["a1", "pool", "--out-root", str(tmp_path / "out"), "--base", "ok", "--n-boot", "100",
                                      "--model", "fake-model"])
    A.main()
    pooled = (tmp_path / "out" / "a1_omni_fake-model_hak_Latn_pooled.md").read_text(encoding="utf-8")
    assert "120 個入聲字、180 個其他字" in pooled and "共同句數 60 句" in pooled and "| bad |" in pooled
    # 多辨識一個生成種子:只辨識新增的(2 個系統 × 30 句),編號 0g2345,同一句的兩次合成先平均
    n_before = len(calls)
    run("sixian", "ok", "bad", extra=("--gen-seeds", "1234,2345"))
    assert calls[n_before:] == [60]
    rows = A.read_tsv_rows(tmp_path / "out" / "sixian" / "transcripts_omni_fake-model_hak_Latn.csv")
    assert {r["k"] for r in rows if r["kind"] == "syn"} == {"0", "0g2345"}
    run("sixian", "ok", "bad", extra=("--gen-seeds", "1234,2345"))
    assert calls[n_before:] == [60]                                   # 再跑一次:不再呼叫 ASR


def test_a1_hanzi_output_is_tokenized_per_character():
    sys.path.insert(0, str(ROOT))
    import a1_asr_check as A
    assert A.tokens("沙人谷甘好") == list("沙人谷甘好") and A.tokens("沙 ngin 谷") == ["沙", "ngin", "谷"]


def test_a1_transcribe_omni_call_shape(monkeypatch):
    """假的 omnilingual_asr:檢查我們呼叫官方介面的方式(模型卡、每個檔一個語言碼、分批、回傳字串)。"""
    import types
    sys.path.insert(0, str(ROOT))
    import a1_asr_check as A
    seen = {"cards": [], "calls": []}

    class FakePipe:
        def __init__(self, model_card=None, **kw):
            seen["cards"].append(model_card)

        def transcribe(self, inp, *, lang=None, batch_size=2):
            seen["calls"].append((list(inp), lang, batch_size))
            assert lang is None or len(lang) == len(inp)
            return [f"hyp:{Path(f).stem}" for f in inp]
    mods = {}
    for name in ("omnilingual_asr", "omnilingual_asr.models", "omnilingual_asr.models.inference", "omnilingual_asr.models.inference.pipeline"):
        mods[name] = types.ModuleType(name)
        monkeypatch.setitem(sys.modules, name, mods[name])
    mods["omnilingual_asr.models.inference.pipeline"].ASRInferencePipeline = FakePipe
    files = [Path(f"/x/{i}.wav") for i in range(5)]
    out = A.transcribe_omni(files, "omniASR_LLM_1B_v2", "hak_Latn", 4, chunk=2)
    assert out == [f"hyp:{i}" for i in range(5)] and seen["cards"] == ["omniASR_LLM_1B_v2"]
    assert [len(c[0]) for c in seen["calls"]] == [2, 2, 1] and all(c[1] == ["hak_Latn"] * len(c[0]) and c[2] == 4 for c in seen["calls"])
    seen["calls"].clear()
    A.transcribe_omni(files, "m", "none", 4)
    assert seen["calls"][0][1] is None
