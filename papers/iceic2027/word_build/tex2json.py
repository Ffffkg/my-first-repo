"""把 ICEIC 論文的 src.tex(+ src.bbl)轉成給 build_docx.js 用的區塊 JSON。只處理這篇論文用到的 LaTeX 寫法。"""
import json
import re
import sys

TEX, BBL, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
src = open(TEX, encoding="utf-8").read()
bbl = open(BBL, encoding="utf-8").read()

# ---- 引用編號:依 .bbl 的順序(IEEEtran 依第一次引用排序)
keys = re.findall(r"\\bibitem\{([^}]+)\}", bbl)
CITE = {k: i + 1 for i, k in enumerate(keys)}
# ---- 圖表編號:依 \label 出現順序
TAB, FIG = {}, {}
for env, lab in re.findall(r"\\begin\{(table|figure)\}.*?\\label\{([^}]+)\}", src, re.S):
    (TAB if env == "table" else FIG)[lab] = len(TAB if env == "table" else FIG) + 1
ROMAN = ["", "I", "II", "III", "IV", "V", "VI", "VII", "VIII", "IX", "X"]

ACCENTS = {r"{\L}": "Ł", r"{\'n}": "ń", r"{\'E}": "É", r"{\'e}": "é", r"\L": "Ł"}
MATH = {r"\pm": "±", r"\rho": "ρ", r"\Delta": "Δ", r"\uparrow": "↑", r"\downarrow": "↓", "-": "−", "+": "+", "=": " = "}


def braced(s, i):
    """s[i] == '{':回傳 (內容, 結束後的位置)。"""
    depth, j = 0, i
    while j < len(s):
        if s[j] == "{":
            depth += 1
        elif s[j] == "}":
            depth -= 1
            if depth == 0:
                return s[i + 1:j], j + 1
        j += 1
    raise ValueError("unbalanced braces: " + s[i:i + 40])


def math_runs(m, fmt):
    """$...$ 的內容 → runs。"""
    out, i = [], 0
    while i < len(m):
        c = m[i]
        if m.startswith("\\", i):
            name = re.match(r"\\[A-Za-z]+", m[i:]).group(0)
            sym = MATH.get(name, "")
            out.append(dict(fmt, text=sym, i=(name == r"\rho") or fmt.get("i", False)))
            i += len(name)
        elif c in "_^":
            i += 1
            if m[i] == "{":
                inner, i = braced(m, i)
            else:
                inner, i = m[i], i + 1
            out.append(dict(fmt, text=inner.replace("-", "−"), **({"sub": True} if c == "_" else {"sup": True})))
        elif c.isalpha():
            out.append(dict(fmt, text=c, i=True))
            i += 1
        elif c == "-":
            out.append(dict(fmt, text="−"))
            i += 1
        elif c == "=":
            out.append(dict(fmt, text=" = "))
            i += 1
        elif c in "{}":
            i += 1
        else:
            out.append(dict(fmt, text=c))
            i += 1
    return out


def inline(s, fmt=None):
    """LaTeX 行內文字 → [{text, b, i, sup, sub, sc}]。"""
    fmt = fmt or {}
    runs, buf, i = [], [], 0

    def flush():
        if buf:
            runs.append(dict(fmt, text="".join(buf)))
            buf.clear()

    while i < len(s):
        c = s[i]
        if s.startswith("---", i):
            buf.append("—"); i += 3; continue
        if s.startswith("--", i):
            buf.append("–"); i += 2; continue
        if s.startswith("``", i):
            buf.append("“"); i += 2; continue
        if s.startswith("''", i):
            buf.append("”"); i += 2; continue
        if c == "'":
            buf.append("’"); i += 1; continue
        if c == "~":
            buf.append("\u00a0"); i += 1; continue
        if c == "$":
            j = s.index("$", i + 1)
            flush()
            runs += math_runs(s[i + 1:j], fmt)
            i = j + 1
            continue
        acc = next((a for a in ACCENTS if s.startswith(a, i)), None)
        if acc:
            buf.append(ACCENTS[acc]); i += len(acc); continue
        if s.startswith("\\'", i):                    # \'x(不帶大括號)
            buf.append({"e": "é", "E": "É", "n": "ń"}.get(s[i + 2], s[i + 2])); i += 3; continue
        if c == "{":
            inner, i = braced(s, i)
            flush()
            runs += inline(inner, fmt)
            continue
        if c == "}":
            i += 1; continue
        if c == "\\":
            if s.startswith("\\,", i):
                buf.append("\u00a0"); i += 2; continue
            if s.startswith("\\ ", i):
                buf.append(" "); i += 2; continue
            if s.startswith("\\%", i) or s.startswith("\\&", i) or s.startswith("\\_", i):
                buf.append(s[i + 1]); i += 2; continue
            if s.startswith("\\\\", i):
                buf.append(" "); i += 2; continue
            for acc, ch in ACCENTS.items():
                if s.startswith(acc, i):
                    buf.append(ch); i += len(acc); break
            else:
                name = re.match(r"\\[A-Za-z]+", s[i:]).group(0)
                i += len(name)
                if name in (r"\textbf", r"\textit", r"\emph", r"\textsuperscript", r"\textnormal", r"\texttt", r"\textsc"):
                    inner, i = braced(s, i)
                    flush()
                    f2 = dict(fmt)
                    if name == r"\textbf": f2["b"] = True
                    if name in (r"\textit", r"\emph"): f2["i"] = True
                    if name == r"\textsuperscript": f2["sup"] = True
                    if name == r"\textsc": f2["sc"] = True
                    if name == r"\textnormal": f2 = {k: v for k, v in f2.items() if k not in ("sc",)}
                    if name == r"\texttt": f2["mono"] = True
                    runs += inline(inner, f2)
                elif name == r"\cite":
                    inner, i = braced(s, i)
                    flush()
                    nums = [CITE[k.strip()] for k in inner.split(",")]
                    runs.append(dict(fmt, text=", ".join(f"[{n}]" for n in nums)))
                elif name == r"\ref":
                    inner, i = braced(s, i)
                    flush()
                    runs.append(dict(fmt, text=ROMAN[TAB[inner]] if inner in TAB else str(FIG[inner])))
                elif name == r"\url":
                    inner, i = braced(s, i)
                    buf.append(inner)
                elif name in (r"\newblock",):
                    buf.append(" ")
                else:
                    pass                                    # 其他指令(\nobreak 等)略過
            continue
        if c == "\n":
            buf.append(" "); i += 1; continue
        buf.append(c)
        i += 1
    flush()
    # 合併多餘空白
    out = []
    for r in runs:
        r["text"] = re.sub(r"[ \t]+", " ", r["text"])
        if r["text"]:
            out.append(r)
    return out


def plain(s):
    return "".join(r["text"] for r in inline(s))


blocks = []
title = re.search(r"\\title\{(.*?)\}\n", src, re.S).group(1)
blocks.append({"t": "title", "runs": inline(title)})

# ---- 作者
auth = re.search(r"\\author\{(.*?)\}\n\n\\maketitle", src, re.S).group(1)
authors = []
for part in auth.split(r"\and"):
    name = re.search(r"\\IEEEauthorblockN\{(.*?)\}\n", part, re.S).group(1)
    aff = re.search(r"\\IEEEauthorblockA\{(.*)\}", part, re.S).group(1)
    lines = [inline(l.strip()) for l in aff.split("\\\\")]
    authors.append({"name": inline(name), "lines": lines})
blocks.append({"t": "authors", "authors": authors})

abstract = re.search(r"\\begin\{abstract\}(.*?)\\end\{abstract\}", src, re.S).group(1).strip()
blocks.append({"t": "abstract", "runs": inline(abstract)})
kw = re.search(r"\\begin\{IEEEkeywords\}(.*?)\\end\{IEEEkeywords\}", src, re.S).group(1).strip()
blocks.append({"t": "keywords", "runs": inline(kw)})


def table_block(body):
    cap = re.search(r"\\caption\{(.*?)\}\n", body).group(1)
    lab = re.search(r"\\label\{([^}]+)\}", body).group(1)
    spec = re.search(r"\\begin\{tabular\}\{([^}]*)\}", body).group(1)
    tab = re.search(r"\\begin\{tabular\}\{[^}]*\}(.*?)\\end\{tabular\}", body, re.S).group(1)
    rows, pending = [], {"top": None, "cmid": []}
    for raw in tab.split("\n"):
        line = raw.strip()
        if not line:
            continue
        while line:
            m = re.match(r"\\(toprule|midrule|bottomrule)", line)
            if m:
                kind = m.group(1)
                if kind == "bottomrule":
                    rows[-1]["bottom"] = "thick"
                else:
                    pending["top"] = "thick" if kind == "toprule" else "thin"
                line = line[m.end():].strip()
                continue
            m = re.match(r"\\cmidrule\(lr\)\{(\d+)-(\d+)\}", line)
            if m:
                rows[-1].setdefault("cmid", []).append((int(m.group(1)), int(m.group(2))))
                line = line[m.end():].strip()
                continue
            # 一般列
            cells_src = [c.strip() for c in line.rstrip("\\").rstrip().rstrip("\\").split("&")]
            cells = []
            for c in cells_src:
                mm = re.match(r"\\multicolumn\{(\d+)\}\{([lcr])\}\{(.*)\}$", c)
                if mm:
                    cells.append({"runs": inline(mm.group(3)), "span": int(mm.group(1)), "align": mm.group(2)})
                else:
                    cells.append({"runs": inline(c), "span": 1})
            rows.append({"cells": cells, "top": pending["top"]})
            pending["top"] = None
            line = ""
    return {"t": "table", "num": ROMAN[TAB[lab]], "caption": inline(cap), "spec": spec, "rows": rows}


# ---- 本文
body = src[src.index("\\section{Introduction}"):src.index("\\bibliographystyle")]
sec_n, sub_n = 0, 0
tokens = re.split(r"(\\begin\{(?:figure|table)\}\[H\].*?\\end\{(?:figure|table)\}|\\begin\{itemize\}.*?\\end\{itemize\})", body, flags=re.S)
fig_imgs = iter(["fig1.png", "fig2.png"])
for tok in tokens:
    if tok.startswith("\\begin{table}"):
        blocks.append(table_block(tok)); continue
    if tok.startswith("\\begin{figure}"):
        cap = re.search(r"\\caption\{(.*?)\}\n", tok).group(1)
        lab = re.search(r"\\label\{([^}]+)\}", tok).group(1)
        blocks.append({"t": "figure", "img": next(fig_imgs), "num": FIG[lab], "caption": inline(cap)})
        continue
    if tok.startswith("\\begin{itemize}"):
        for it in re.split(r"\\item\s*", tok.replace("\\begin{itemize}", "").replace("\\end{itemize}", ""))[1:]:
            blocks.append({"t": "bullet", "runs": inline(it.strip())})
        continue
    for para in re.split(r"\n\s*\n", tok):
        para = para.strip()
        if not para:
            continue
        # 一段裡可能有 \section / \subsection / \parhead 開頭
        while para:
            m = re.match(r"\\section\*?\{([^}]*)\}\s*", para)
            if m:
                if para.startswith("\\section*"):
                    blocks.append({"t": "h1", "text": plain(m.group(1)).upper()})
                else:
                    sec_n += 1; sub_n = 0
                    blocks.append({"t": "h1", "text": f"{ROMAN[sec_n]}. {plain(m.group(1)).upper()}"})
                para = para[m.end():]; continue
            m = re.match(r"\\subsection\{([^}]*)\}\s*", para)
            if m:
                sub_n += 1
                blocks.append({"t": "h2", "text": f"{chr(64 + sub_n)}. {plain(m.group(1))}"})
                para = para[m.end():]; continue
            m = re.match(r"\\parhead\{([^}]*)\}\s*", para)
            if m:
                blocks.append({"t": "parhead", "text": plain(m.group(1))})
                para = para[m.end():]; continue
            blocks.append({"t": "p", "runs": inline(para)})
            para = ""

# ---- 參考文獻
refs = []
for k, txt in re.findall(r"\\bibitem\{([^}]+)\}\s*(.*?)(?=\\bibitem|\\end\{thebibliography\})", bbl, re.S):
    refs.append(inline(re.sub(r"\s+", " ", txt.strip())))
blocks.append({"t": "refs", "items": refs})

json.dump(blocks, open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print(f"{len(blocks)} blocks, {len(refs)} refs, cites {len(CITE)}, tables {len(TAB)}, figs {len(FIG)}")
