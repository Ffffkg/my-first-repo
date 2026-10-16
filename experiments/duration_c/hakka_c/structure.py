"""把一句的「模型符號序列」與 MFA TextGrid 對起來,找出每個音節的母音核位置。

做法(不依賴前端程式):
  1. filelist 的 PUA 字元 → 符號名稱(例如 's'、'ii@5'、'd'、'sp')。
  2. 去掉停頓符號後的音素序列,必須與 TextGrid phones 層(去掉靜音)逐一對應,
     名稱也要一致(符號名稱 '@' 之前的部分 == MFA 音素標籤)。不符就整句跳過,不硬湊。
  3. 音節邊界取自 TextGrid words 層(每個 word 就是一個拼音音節,如 siid5)。
  4. 母音核 = 音節內唯一含 a/e/i/o/u 的音素;成音節鼻音(ng11、m11)沒有母音核,跳過。
  5. 入聲 = 拼音以 b/d/g(或 p/t/k)結尾且不是 -ng(與手冊的 is_entering() 相同)。
"""
import json
import re
from dataclasses import dataclass, field

from .config import MFA_INVALID, MFA_SILENCE, PAUSE_SYMBOLS, VOWEL_CHARS
from .textgrid import find_tier


# ---------------------------------------------------------------- PUA 對照
def load_pua_map(vocab_json):
    """從 hakka_matcha_vocab.json 找出「PUA 字元 → 符號名稱」的對照。

    不確定 setup_matcha.py 寫出的 JSON 結構,所以用啟發式搜尋:
    找一個 dict,其中一邊是單一 PUA 字元(U+E000–U+F8FF),另一邊是符號名稱。
    找不到就報錯,請把 probe_env.py 的輸出貼回來。
    """
    data = json.load(open(vocab_json, encoding="utf-8"))
    best = None

    def is_pua(s):
        return isinstance(s, str) and len(s) == 1 and 0xE000 <= ord(s) <= 0xF8FF

    def visit(obj):
        nonlocal best
        if isinstance(obj, dict) and obj:
            vals = list(obj.values())
            keys = list(obj.keys())
            if sum(map(is_pua, vals)) >= 0.9 * len(vals):          # name -> pua
                cand = {v: k for k, v in obj.items() if is_pua(v)}
            elif sum(map(is_pua, keys)) >= 0.9 * len(keys):       # pua -> name
                cand = {k: v for k, v in obj.items() if is_pua(k) and isinstance(v, str)}
            else:
                cand = None
            if cand and (best is None or len(cand) > len(best)):
                best = cand
            for v in vals:
                visit(v)
        elif isinstance(obj, list):
            # 也可能是 [[name, pua], ...] 或 [{"symbol":..,"pua":..}, ...]
            pairs = {}
            for it in obj:
                if isinstance(it, (list, tuple)) and len(it) == 2:
                    a, b = it
                    if is_pua(b) and isinstance(a, str):
                        pairs[b] = a
                    elif is_pua(a) and isinstance(b, str):
                        pairs[a] = b
                elif isinstance(it, dict):
                    pu = [v for v in it.values() if is_pua(v)]
                    nm = [v for v in it.values() if isinstance(v, str) and not is_pua(v)]
                    if len(pu) == 1 and nm:
                        pairs[pu[0]] = nm[0]
                visit(it)
            if pairs and (best is None or len(pairs) > len(best)):
                best = pairs

    visit(data)
    if not best:
        raise ValueError(f"在 {vocab_json} 找不到 PUA 對照表;請執行 probe_env.py 並把輸出貼回來")
    return best


def base_of(name):
    """'ii@5' -> 'ii';'s' -> 's'。"""
    return name.split("@", 1)[0]


def is_pause(name):
    return base_of(name) in PAUSE_SYMBOLS


def has_vowel(phone):
    return any(c in VOWEL_CHARS for c in phone)


_SYL = re.compile(r"^([a-z]+)(\d+)$")


def is_entering_syllable(word):
    m = _SYL.match(word.lower())
    if not m:
        return None
    base = m.group(1)
    if base.endswith("ng"):
        return False
    return base.endswith(("b", "d", "g", "p", "t", "k"))


# ---------------------------------------------------------------- 結構
@dataclass
class Syllable:
    idx: int                 # 句中第幾個音節(0 起)
    word: str                # 拼音,如 siid5
    sym_idx: list            # 該音節各音素在「非 blank 符號序列」中的位置
    nuc: int                 # 母音核的符號位置
    entering: bool
    coda: str                # 'b'/'d'/'g'/'ng'/'n'/'m'/''(無韻尾)
    ref_ms: float            # MFA 母音核時長(ms)
    prepausal: bool          # 音節後面緊接停頓或句末
    tone: str


@dataclass
class Utt:
    stem: str
    names: list              # 非 blank 符號名稱,長度 n
    phone_sym: list          # 第 k 個 MFA 音素對應的符號位置
    phone_iv: list           # 第 k 個 MFA 音素的 (start_s, end_s),原始 TextGrid 座標
    syllables: list = field(default_factory=list)


class StructError(Exception):
    pass


def build_utt(stem, names, tiers):
    """names: 非 blank 的符號名稱序列;tiers: read_textgrid() 的結果。"""
    phones = find_tier(tiers, "phones")
    words = find_tier(tiers, "words")
    if any(t.strip().lower() in MFA_INVALID for _, _, t in phones):
        raise StructError("MFA 有 spn/<unk>")
    ph = [(a, b, t.strip()) for a, b, t in phones if t.strip().lower() not in MFA_SILENCE]
    sym_ph = [i for i, nm in enumerate(names) if not is_pause(nm)]
    if len(ph) != len(sym_ph):
        raise StructError(f"音素數不符:符號 {len(sym_ph)} vs MFA {len(ph)}")
    for k, (i, (_, _, lab)) in enumerate(zip(sym_ph, ph)):
        if base_of(names[i]) != lab:
            raise StructError(f"第 {k} 個音素名稱不符:符號 {names[i]} vs MFA {lab}")

    utt = Utt(stem, list(names), sym_ph, [(a, b) for a, b, _ in ph])
    k = 0
    syl_i = 0
    eps = 1e-4
    for wa, wb, wt in words:
        w = wt.strip()
        if w.lower() in MFA_SILENCE:
            continue
        members = []
        while k < len(ph) and ph[k][0] >= wa - eps and ph[k][1] <= wb + eps:
            members.append(k)
            k += 1
        if not members:
            raise StructError(f"word {w} 內沒有音素")
        ent = is_entering_syllable(w)
        vow = [m for m in members if has_vowel(ph[m][2])]
        if ent is not None and len(vow) == 1:
            m = vow[0]
            nuc = sym_ph[m]
            last = ph[members[-1]][2]
            coda = last if members[-1] != m else ""
            after = sym_ph[members[-1]] + 1
            prepausal = after >= len(names) or is_pause(names[after])
            mm = _SYL.match(w.lower())
            utt.syllables.append(Syllable(
                idx=syl_i, word=w, sym_idx=[sym_ph[x] for x in members], nuc=nuc,
                entering=bool(ent), coda=coda, ref_ms=(ph[m][1] - ph[m][0]) * 1000.0,
                prepausal=prepausal, tone=mm.group(2) if mm else ""))
        syl_i += 1
    if k != len(ph):
        raise StructError(f"有 {len(ph) - k} 個音素不在任何 word 內")
    return utt
