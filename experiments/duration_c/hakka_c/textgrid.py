"""Praat TextGrid 讀取(長格式與短格式都支援),不依賴任何套件。"""
import re

_STR = r'"((?:[^"]|"")*)"'


def _unq(s):
    return s.replace('""', '"')


def _read(path):
    raw = open(path, "rb").read()
    for enc in ("utf-8-sig", "utf-16", "latin-1"):
        try:
            return raw.decode(enc)
        except UnicodeDecodeError:
            continue
    raise UnicodeDecodeError("cannot decode " + str(path))


def read_textgrid(path):
    """回傳 {tier_name: [(xmin, xmax, text), ...]},只收 IntervalTier。"""
    txt = _read(path)
    if re.search(r"^\s*item\s*\[", txt, re.M):
        return _read_long(txt)
    return _read_short(txt)


def _read_long(txt):
    tiers, cur, iv, is_interval_tier = {}, None, None, False
    for line in txt.splitlines():
        s = line.strip()
        if s.startswith("item ["):
            cur, iv, is_interval_tier = None, None, False
        elif s.startswith("class ="):
            is_interval_tier = "IntervalTier" in s
        elif s.startswith("name =") and is_interval_tier:
            cur = _unq(re.search(_STR, s).group(1))
            tiers[cur] = []
        elif s.startswith("intervals [") and cur is not None:
            iv = {}
        elif iv is not None and cur is not None:
            if s.startswith("xmin ="):
                iv["xmin"] = float(s.split("=", 1)[1])
            elif s.startswith("xmax ="):
                iv["xmax"] = float(s.split("=", 1)[1])
            elif s.startswith("text ="):
                m = re.search(_STR, s)
                iv["text"] = _unq(m.group(1)) if m else ""
                tiers[cur].append((iv["xmin"], iv["xmax"], iv["text"]))
                iv = None
    return tiers


def _read_short(txt):
    toks = re.findall(_STR + r"|([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)", txt)
    toks = [("s", _unq(a)) if b == "" else ("n", float(b)) for a, b in toks]
    # header: "ooTextFile" "TextGrid" xmin xmax <exists> size
    i = 0
    while i < len(toks) and not (toks[i][0] == "s" and toks[i][1] in ("IntervalTier", "TextTier")):
        i += 1
    tiers = {}
    while i < len(toks):
        cls = toks[i][1]
        name = toks[i + 1][1]
        n = int(toks[i + 4][1])
        i += 5
        items = []
        step = 3 if cls == "IntervalTier" else 2
        for _ in range(n):
            if cls == "IntervalTier":
                items.append((toks[i][1], toks[i + 1][1], toks[i + 2][1]))
            i += step
        if cls == "IntervalTier":
            tiers[name] = items
    return tiers


def find_tier(tiers, kind):
    """kind = 'words' 或 'phones'。MFA 的 tier 名稱可能是 'phones' 或 'XF - phones'。"""
    for name in tiers:
        if name == kind:
            return tiers[name]
    for name in tiers:
        if name.lower().endswith(kind):
            return tiers[name]
    raise KeyError(f"no tier ending with '{kind}' in {list(tiers)}")
