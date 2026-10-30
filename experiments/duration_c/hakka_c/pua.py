"""找出「PUA 字元 → 音素名稱」的對照表。

依序嘗試(每個候選都要先通過 TextGrid 驗證,才採用):
  1. 環境變數 HAKKA_VOCAB 指定的 JSON
  2. benchmark/hakka_matcha_vocab.json、benchmark/*.json、其他名稱含 vocab 的 JSON
  3. 之前學好的 exp_c/pua_map_learned.json
  4. 直接從資料學(見 learn_pua_map):先推斷哪些字元是停頓,去掉後與 MFA phones 層逐一對照,
     多數決決定每個字元代表哪個音素;一致率 ≥ 97% 才採用。結果存成 exp_c/pua_map_learned.json 供檢查。

注意:學到的名稱是 MFA 音素標籤(例如 'ii',不含 @聲調);程式只用「音素名稱 + 是否停頓」,
聲調與是否入聲取自 TextGrid words 層的拼音,所以不影響結果。
"""
import json
import os
import random
from collections import Counter, defaultdict
from pathlib import Path

from . import config as C
from .structure import StructError, build_utt, load_pua_map
from .textgrid import find_tier, read_textgrid

LEARNED = C.OUT_ROOT / "pua_map_learned.json"


def _clean(text, cleaners=("hakka_cleaners",)):
    import sys
    if str(C.MATCHA) not in sys.path:
        sys.path.insert(0, str(C.MATCHA))
    from matcha.text import text_to_sequence
    return text_to_sequence(text, list(cleaners))[1]


def _sample_lines(n, seed=C.SPLIT_SEED):
    out = []
    for d in C.DIALECTS:
        for split in ("test", "val"):
            fl = C.filelist("joint", d, split)
            if fl.exists():
                out += [(d, l.rstrip("\n").split("|")) for l in open(fl, encoding="utf-8") if l.strip()]
    random.Random(seed).shuffle(out)
    return out[:n]


_TG = {}


def _tg_index(d):
    if d not in _TG:
        _TG[d] = {p.stem: p for p in C.textgrid_dir(d).rglob("*.TextGrid")}
    return _TG[d]


def score_map(pmap, n=60, cleaners=("hakka_cleaners",)):
    """在 n 句上檢查對照表:回傳結構對應成功的比例。"""
    ok = tot = 0
    for d, (wav, _, text) in _sample_lines(n, seed=1):
        stem = Path(wav).stem
        tg = _tg_index(d).get(stem)
        if tg is None:
            continue
        tot += 1
        try:
            build_utt(stem, [pmap[c] for c in _clean(text, cleaners)], read_textgrid(tg))
            ok += 1
        except (KeyError, StructError):
            pass
    return ok / max(tot, 1)


def learn_pua_map(n=1500, cleaners=("hakka_cleaners",), min_purity=0.97):
    """1. 找停頓字元集合 P:每句「字元數 − MFA 音素數」= 該句停頓字元個數,
          且停頓字元在任何一句出現次數都不會超過這個差。從候選中選出讓等式成立句數最多的組合。
       2. 去掉 P 之後每句長度相等,逐位置配對,多數決 + 一致率 ≥ 97% 決定每個字元的音素。"""
    from itertools import combinations
    data = []
    for d, (wav, _, text) in _sample_lines(n):
        tg = _tg_index(d).get(Path(wav).stem)
        if tg is None:
            continue
        chars = _clean(text, cleaners)
        ph = [t.strip() for _, _, t in find_tier(read_textgrid(tg), "phones")
              if t.strip().lower() not in C.MFA_SILENCE]
        if any(p.lower() in C.MFA_INVALID for p in ph) or len(chars) < len(ph):
            continue
        data.append((chars, ph, Counter(chars), len(chars) - len(ph)))
    if not data:
        raise ValueError("沒有可用來學習的句子(找不到 filelist 或 TextGrid)")
    seen = Counter(c for ch, _, _, _ in data for c in set(ch))
    cand = [c for c in seen if all(cnt[c] <= diff for _, _, cnt, diff in data)]
    cand = sorted(cand, key=lambda c: -seen[c])[:8]

    def fit(P):
        return sum(sum(cnt[c] for c in P) == diff for _, _, cnt, diff in data) / len(data)

    def mapping(P):
        pair = defaultdict(Counter)
        used = 0
        for chars, ph, _, _ in data:
            rest = [c for c in chars if c not in P]
            if len(rest) == len(ph):
                used += 1
                for c, p in zip(rest, ph):
                    pair[c][p] += 1
        tot = sum(sum(v.values()) for v in pair.values())
        purity = sum(v.most_common(1)[0][1] for v in pair.values()) / max(tot, 1)
        return pair, used, purity

    combos = [()] + [P for r in range(1, min(4, len(cand)) + 1) for P in combinations(cand, r)]
    fits = {P: fit(P) for P in combos}
    best_fit = max(fits.values())
    # 長度差解釋得一樣好的組合中,選「去掉後字元↔音素最一致」的那組(同分取較小的集合)
    near = [P for P in combos if fits[P] >= best_fit - 0.02]
    scored = [(mapping(set(P))[2], -len(P), P) for P in near]
    best = max(scored)[2]
    P = set(best)
    pair, used, _ = mapping(P)
    pmap, bad = {c: "sp" for c in P}, {}
    for c, cnt in pair.items():
        p, k = cnt.most_common(1)[0]
        if k / sum(cnt.values()) >= min_purity:
            pmap[c] = p
        else:
            bad[f"U+{ord(c):04X}"] = dict(cnt.most_common(3))
    best_fit = fits[best]
    info = {"chars": len(pmap), "pause_chars": [f"U+{ord(c):04X}" for c in sorted(P)],
            "pause_fit": round(best_fit, 3), "utts_used": used, "utts_total": len(data), "low_purity": bad}
    return pmap, info


def resolve_pua_map(cleaners=("hakka_cleaners",), verbose=True, min_score=0.8):
    cands = []
    if os.environ.get("HAKKA_VOCAB"):
        cands.append(Path(os.environ["HAKKA_VOCAB"]).expanduser())
    cands += [C.VOCAB_JSON] + sorted(C.BENCH.glob("*.json"))
    cands += sorted(C.HAKKA.glob("*/*vocab*.json")) + sorted(C.MATCHA.glob("data/hakka/**/*vocab*.json"))
    seen = set()
    for p in cands:
        if p in seen or not p.exists():
            continue
        seen.add(p)
        try:
            m = load_pua_map(p)
        except Exception:  # noqa: BLE001
            continue
        s = score_map(m, cleaners=cleaners)
        if verbose:
            print(f"  候選對照表 {p}:{len(m)} 個字元,結構對應成功率 {s:.0%}")
        if s >= min_score:
            return m, str(p)
    if LEARNED.exists():
        m = {k: v for k, v in json.load(open(LEARNED, encoding="utf-8"))["map"].items()}
        if score_map(m, cleaners=cleaners) >= min_score:
            return m, str(LEARNED)
    if verbose:
        print("  找不到可用的對照表 JSON,改從 filelist + TextGrid 學習…")
    m, info = learn_pua_map(cleaners=cleaners)
    s = score_map(m, cleaners=cleaners)
    LEARNED.parent.mkdir(parents=True, exist_ok=True)
    json.dump({"map": m, "info": info, "score": s,
               "readable": {f"U+{ord(k):04X}": v for k, v in sorted(m.items())}},
              open(LEARNED, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    if verbose:
        print(f"  學到 {info['chars']} 個字元;停頓字元 {info['pause_chars']}(解釋 {info['pause_fit']:.0%} 句的長度差);"
              f"結構對應成功率 {s:.0%}"
              f";一致率不足的字元:{info['low_purity'] or '無'} → 已存 {LEARNED}")
    if s < min_score:
        raise ValueError(f"學到的對照表只對上 {s:.0%} 的句子,請把 {LEARNED} 與 probe_report.txt 貼回對話")
    return m, f"learned ({LEARNED})"
