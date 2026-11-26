"""客語符號序列 → 音節分組(純 Python,不依賴 torch;模型與檢查程式共用)。

文字前端把每個音節寫成「聲母?+ 帶聲調的韻母(如 a@24、iau@11)+ 韻尾?」,序列中沒有音節分隔符號,
但每個音節剛好有一個「錨點」:帶聲調的符號(含成音節鼻音 ng@11 等)或含母音的符號。
相鄰兩個錨點之間的子音依下列規則分給前後音節:
  0 個:無
  ≥2 個:第一個是前一音節的韻尾,其餘是後一音節的聲母
  1 個:前一音節是入聲(調值 2、5,一定有塞音韻尾)→ 韻尾;
       是鼻音 m/n/ng → 依 nasal_rule(「onset」= 當後一音節的聲母,「coda」= 當前一音節的韻尾);
       其他 → 聲母
停頓(sp、sil)切開音節;停頓前的子音歸前一音節,停頓後的子音歸後一音節。
blank(id 0)若前後兩個符號屬於同一音節,就歸該音節,否則不屬於任何音節(-1)。
"""

PAUSES = {"sp", "sil"}
NASALS = {"m", "n", "ng"}
CHECKED_TONES = {"2", "5"}
VOWELS = set("aeiou")


def classify(name):
    """回傳 (類別, 是否入聲調, 是否鼻音);類別為 'pause' / 'anchor' / 'cons'。"""
    base, _, tone = name.partition("@")
    if base in PAUSES:
        return "pause", False, False
    if tone or any(c in VOWELS for c in base):
        return "anchor", tone in CHECKED_TONES, False
    return "cons", False, base in NASALS


def segment(ids, names, nasal_rule="onset"):
    """ids:符號 id 序列(0 = blank/pad,只傳有效長度);names[id] = 符號名稱(names[0] 不用)。
    回傳 (每個位置的音節編號(-1 = 不屬於音節), 每個音節是否在停頓或句末之前)。"""
    L = len(ids)
    syl = [-1] * L
    toks = [(p, classify(names[i])) for p, i in enumerate(ids) if i != 0]
    prepausal = []
    n = 0
    k = 0
    while k < len(toks):                         # 逐段處理(以停頓切開)
        if toks[k][1][0] == "pause":
            k += 1
            continue
        seg = []
        while k < len(toks) and toks[k][1][0] != "pause":
            seg.append(toks[k])
            k += 1
        anchors = [j for j, (_, c) in enumerate(seg) if c[0] == "anchor"]
        if not anchors:
            continue
        sid = {a: n + m for m, a in enumerate(anchors)}
        for a in anchors:
            syl[seg[a][0]] = sid[a]
        for j in range(anchors[0]):              # 第一個錨點之前 → 聲母
            syl[seg[j][0]] = sid[anchors[0]]
        for j in range(anchors[-1] + 1, len(seg)):   # 最後一個錨點之後 → 韻尾
            syl[seg[j][0]] = sid[anchors[-1]]
        for a, b in zip(anchors, anchors[1:]):
            run = list(range(a + 1, b))
            if not run:
                continue
            if len(run) >= 2:
                syl[seg[run[0]][0]] = sid[a]
                for j in run[1:]:
                    syl[seg[j][0]] = sid[b]
                continue
            j = run[0]
            checked = seg[a][1][1]
            nasal = seg[j][1][2]
            to_prev = checked or (nasal and nasal_rule == "coda")
            syl[seg[j][0]] = sid[a] if to_prev else sid[b]
        prepausal += [False] * (len(anchors) - 1) + [True]
        n += len(anchors)
    for p in range(L):                           # blank:前後同音節才歸入
        if ids[p] == 0 and 0 < p < L - 1 and syl[p - 1] >= 0 and syl[p - 1] == syl[p + 1]:
            syl[p] = syl[p - 1]
    return syl, prepausal


def segment_from_groups(ids, groups):
    """音節分組由文字前端提供(拼音本來就知道音節邊界)時使用。
    ids:含 blank 的 id 序列;groups:每個非 blank 符號的音節編號(0 起連續,停頓 = -1)。
    回傳格式與 segment() 相同。"""
    L = len(ids)
    nb = [p for p, i in enumerate(ids) if i != 0]
    if len(nb) != len(groups):
        raise ValueError(f"符號數 {len(nb)} 與分組長度 {len(groups)} 不符")
    syl = [-1] * L
    for p, g in zip(nb, groups):
        syl[p] = g
    for p in range(L):
        if ids[p] == 0 and 0 < p < L - 1 and syl[p - 1] >= 0 and syl[p - 1] == syl[p + 1]:
            syl[p] = syl[p - 1]
    n = max(groups) + 1 if any(g >= 0 for g in groups) else 0
    last = {}
    for j, g in enumerate(groups):
        if g >= 0:
            last[g] = j
    prepausal = [last[k] + 1 >= len(groups) or groups[last[k] + 1] < 0 for k in range(n)]
    return syl, prepausal


def table_key(ids, pua_start=0xE000):
    """非 blank 的符號 id → 與 filelist 文字相同的私用區字串(查表用)。"""
    return "".join(chr(pua_start + i - 1) for i in ids if i != 0)
