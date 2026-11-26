"""A1:用語音辨識(ASR)模型當「獨立於 MFA」的客觀檢查——合成語音能不能被辨識得跟真人一樣,入聲字有沒有特別吃虧。

為什麼:我們所有時長指標都靠 MFA 量。這裡改用一個完全不同的系統(Meta Omnilingual ASR,有 hak_Latn 客語;
也可選 Whisper 客語微調)去「聽」合成語音,看辨識結果跟聽真人錄音的結果差多少。

試跑看到的事實(omniASR_LLM_1B_v2 + hak_Latn):輸出是白話字(Pha̍k-fa-sṳ)拼寫,音節間用連字號,有聲調符號
(例:pûn-ngìn kit-tó),入聲韻尾寫 -p/-t/-k(我們的客語拼音是 -b/-d/-g);有些句子冒出漢字、韓文、泰文、阿拉伯數字(辨識幻覺),
少數句子整句都是非拉丁字母。所以所有指標都是「ASR(合成)對 ASR(同一句真人錄音)」,不依賴 ASR 的拼寫與標準答案一致;
非拉丁字母的字一律忽略,真人辨識結果幾乎沒有拉丁字母的句子整句排除(否則 CER 會被極小的分母炸掉)。
(曾經試過「對標準答案的 CER」:拼寫不同,真人錄音的底線比合成語音還差,沒有資訊量,已拿掉。)

指標(報告最前面有一張各系統與 base 的差的摘要表):
  1. 一致度 CER:ASR(合成)對 ASR(真人)的字元錯誤率,全部句子加總後相除。字母層級 = 忽略聲調符號;含聲調符號 = 聲調符號也算。
  2. 入聲字 vs 其他字:入聲字 = 真人辨識結果裡韻尾是 -p/-t/-k 的字。把兩邊的辨識結果逐字對齊,看
       (a) 字的相似度與「入聲 − 其他字」的差距,(b) 入聲保留率(真人聽成入聲的字,合成語音也聽成入聲)與其他字誤成入聲率。
  3. 聲調記號一致率:對得上的字之中,聲調記號(^ ` ´ ̍ 或無)與真人相同的比例。

    python a1_asr_check.py run --dialect sixian --backend omni --model omniASR_LLM_1B_v2 \\
        --sys base=$E/base_sixian_step40000,... --sys c6=... --out-dir ~/hakka_tts/exp_c/a1/sixian
    python a1_asr_check.py analyze --dialect sixian --out-dir ~/hakka_tts/exp_c/a1/sixian   # 只重算報告(模型、語言設定要與 run 相同)
    python a1_asr_check.py pool --out-root ~/hakka_tts/exp_c/a1                              # 合併兩個腔調一份報告(句子數加倍)

生成種子:預設每個訓練種子只辨識第一個生成種子(1234)的合成音檔;--gen-seeds 1234,2345,3456 可以多辨識幾個(已辨識的跳過),
同一句的多次合成會先平均,信賴區間比較窄。
辨識結果存在 out-dir/transcripts_<後端>_<模型>_<語言>.csv,可續跑;換模型或語言設定會寫到另一個檔案,不會混用。

需要的環境:omni 後端要 omnilingual-asr(見 a1_setup_env.sh,建議獨立的 conda 環境 asr);報告部分只用 numpy。
"""
import argparse
import csv
import math
import re
import sys
import unicodedata
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from hakka_c import config as C  # noqa: E402
from hakka_c.stats import boot_idx  # noqa: E402
from hakka_c.textgrid import find_tier, read_textgrid  # noqa: E402

PUNCT = re.compile(r"[\s\-–—_.,;:!?\"'“”‘’()\[\]{}<>…、，。！？；：·/\\|]+")
CJK = "\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff"


# ------------------------------------------------------------------ 字串處理
def lev(a, b):
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i]
        for j, y in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (x != y)))
        prev = cur
    return prev[-1]


def tokens(s):
    """以空白 / 標點切詞;漢字一字一個 token(客語 Whisper 可能輸出漢字)。"""
    s = unicodedata.normalize("NFKC", s).lower()
    out = []
    for t in PUNCT.split(s):
        out += re.findall(rf"[{CJK}]|[^{CJK}]+", t)
    return [t for t in out if t]


def strip_marks(s):
    d = unicodedata.normalize("NFD", s)
    return unicodedata.normalize("NFC", "".join(c for c in d if unicodedata.category(c) != "Mn"))


def letters(s):
    """忽略聲調符號與數字,只留 a–z(與漢字)。"""
    return re.sub(rf"[^a-z{CJK}]", "", strip_marks(unicodedata.normalize("NFKC", s).lower()))


def with_marks(s):
    """保留聲調符號(組合記號)與數字,去掉空白與標點。"""
    d = unicodedata.normalize("NFD", unicodedata.normalize("NFKC", s).lower())
    return "".join(c for c in d if c.isalnum() or unicodedata.category(c) == "Mn")


TONE_MARKS = set("\u0300\u0301\u0302\u0303\u0304\u030c\u030d")   # 重音 ` 、銳音 ´ 、揚抑符 ^ 、~ 、長音符 ¯ 、ˇ 、上方豎線 ̍(ṳ 的下加分音符 U+0324 是字母的一部分,不算)
MATCH_SIM = 0.5                       # 對齊後字母相似度達此值,才算「對得上」的字
COUNT_KEYS = ("n_et", "n_non", "sim_et", "sim_non", "m_et", "m_non", "tone_et", "tone_non", "kept_et", "false_non")


def tone_of(tok):
    """token 的聲調記號(只認 TONE_MARKS);沒有記號 = ''。"""
    d = unicodedata.normalize("NFD", unicodedata.normalize("NFKC", tok).lower())
    return "".join(sorted(c for c in d if c in TONE_MARKS))


def latin_tokens(s):
    """辨識輸出 → [(a–z 字母, 聲調記號)];只留拉丁字母的 token(數字、漢字、韓文等辨識幻覺忽略)。"""
    return [(let, tone_of(t)) for t in tokens(s) for let in [letters(t)] if re.fullmatch(r"[a-z]+", let)]


def ends_stop(let):
    """白話字的入聲:韻尾 -p / -t / -k。"""
    return let[-1] in "ptk"


def align_pairs(ref, hyp):
    """ref、hyp:token 字串列。回傳每個 ref token 一筆 (對到的 hyp 位置 或 None, 相似度 0–1);被刪掉 = (None, 0.0)。"""
    n, m = len(ref), len(hyp)
    D = [[0.0] * (m + 1) for _ in range(n + 1)]
    B = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        D[i][0], B[i][0] = float(i), 2
    for j in range(1, m + 1):
        D[0][j], B[0][j] = float(j), 3
    sub_cost = {}
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            c = lev(ref[i - 1], hyp[j - 1]) / max(len(ref[i - 1]), len(hyp[j - 1]), 1)
            sub_cost[i, j] = c
            opts = ((D[i - 1][j - 1] + c, 1), (D[i - 1][j] + 1.0, 2), (D[i][j - 1] + 1.0, 3))
            D[i][j], B[i][j] = min(opts, key=lambda t: t[0])
    out = [(None, 0.0)] * n
    i, j = n, m
    while i > 0 or j > 0:
        b = B[i][j]
        if b == 1:
            out[i - 1] = (j - 1, 1.0 - sub_cost[i, j])
            i, j = i - 1, j - 1
        elif b == 2:
            i -= 1
        else:
            j -= 1
    return out


def align_tokens(ref, hyp):
    """每個 ref token 的相似度(0–1,被刪掉 = 0)。"""
    return [sim for _, sim in align_pairs(ref, hyp)]


# ------------------------------------------------------------------ 逐句指標與報告
RELIABLE_FRAC = 0.5      # 真人辨識結果的拉丁字母字數,至少要有「標準答案音節數 × 此比例」,否則整句排除(多半是辨識幻覺,例如整句輸出泰文)


def latin_marked(s):
    """辨識輸出 → 拉丁字母 token 串接(保留聲調記號),用來算含聲調符號的 CER。"""
    return "".join(with_marks(t) for t in tokens(s) if re.fullmatch(r"[a-z]+", letters(t)))


def reliable(real, n_ref):
    """真人錄音的辨識結果可不可靠:拉丁字母的字數要夠(整句被辨識成泰文、漢字等就不可靠,當作參考答案會讓 CER 爆表)。"""
    return len(latin_tokens(real)) >= RELIABLE_FRAC * max(n_ref, 4)


def token_counts(real, hyp):
    """真人與合成語音的辨識結果逐字對齊,數入聲 / 非入聲字的相似度、入聲保留、聲調記號一致(COUNT_KEYS)。
    入聲字 = 真人辨識結果裡韻尾是 -p / -t / -k 的字。"""
    rt, ht = latin_tokens(real), latin_tokens(hyp)
    cnt = dict.fromkeys(COUNT_KEYS, 0.0)
    for i, (j, sim) in enumerate(align_pairs([x[0] for x in rt], [x[0] for x in ht])):
        c = "et" if ends_stop(rt[i][0]) else "non"
        cnt[f"n_{c}"] += 1
        cnt[f"sim_{c}"] += sim
        if j is not None and sim >= MATCH_SIM:
            cnt[f"m_{c}"] += 1
            cnt[f"tone_{c}"] += rt[i][1] == ht[j][1]
            if c == "et":
                cnt["kept_et"] += ends_stop(ht[j][0])
            else:
                cnt["false_non"] += ends_stop(ht[j][0])
    return cnt


def per_stem_metrics(real_hyp, ref_n, syn_hyp):
    """real_hyp: {stem: 真人辨識結果};ref_n: {stem: 標準答案音節數};syn_hyp: {(系統, 種子, stem): 合成語音辨識結果}。
    回傳 {系統: {指標: {stem: 各種子平均}}};真人辨識結果不可靠的句子不收。"""
    acc = {}
    for (sysname, k, stem), hyp in syn_hyp.items():
        real = real_hyp.get(stem, "")
        if not reliable(real, ref_n.get(stem, 0)):
            continue
        lr, lh = "".join(l for l, _ in latin_tokens(real)), "".join(l for l, _ in latin_tokens(hyp))
        mr, mh = latin_marked(real), latin_marked(hyp)
        m = {"ed_letters": lev(lh, lr), "len_letters": len(lr), "ed_marks": lev(mh, mr), "len_marks": len(mr)}
        m.update(token_counts(real, hyp))
        for key, v in m.items():
            acc.setdefault(sysname, {}).setdefault(key, {}).setdefault(stem, []).append(v)
    return {s: {key: {st: float(np.mean(v)) for st, v in d.items()} for key, d in ms.items()} for s, ms in acc.items()}


def ci_str(point, draws, scale=100.0, digits=1):
    lo, hi = np.percentile(draws, [2.5, 97.5])
    return f"{scale * point:.{digits}f} [{scale * lo:.{digits}f}, {scale * hi:.{digits}f}]"


def diff_str(point, draws, scale=100.0, digits=1):
    lo, hi = np.percentile(draws, [2.5, 97.5])
    return f"{scale * point:+.{digits}f} [{scale * lo:+.{digits}f}, {scale * hi:+.{digits}f}]"


def build_report(M, order, base, n_boot=5000, n_excluded=0):
    """M: per_stem_metrics 的結果。order:系統顯示順序;base:比較基準的系統名稱;n_excluded:因真人辨識不可靠而排除的句數。"""
    L = []
    order = [s for s in order if s in M]
    if not order:
        return "(沒有任何可用的辨識結果:真人的辨識結果不可靠,或合成語音沒有辨識結果。)"
    common = sorted(set.intersection(*[set(M[s]["ed_letters"]) for s in order]))
    if not common:
        return "(各系統沒有共同的句子可比較。)"
    ex = f"(另有 {n_excluded} 句因為真人錄音的辨識結果幾乎不是拉丁字母〔多半是辨識幻覺〕而排除)" if n_excluded else ""
    L += [f"共同句數 {len(common)} 句{ex}。CI = 以句子重抽 {n_boot} 次的 95% 區間;「與 {base} 的差」為同一批句子的配對差。\n"]
    if "copy" in order:
        L += ["`copy` = 真人錄音的真實 mel 直接過同一個 HiFi-GAN 重建(聲學模型與時長都完美時的天花板);其他系統要跟它比,才知道還有多少進步空間。\n"]
    idx = boot_idx(len(common), n_boot)

    def arr(s, key, stems=common):
        return np.array([M[s][key][t] for t in stems])

    def ratio(s, nums, dens):
        """比例 = 分子總和 / 分母總和(跨句子加總);回傳 (點估計, 句子重抽的 draws)。"""
        n = sum(arr(s, k) for k in ((nums,) if isinstance(nums, str) else nums))
        d = sum(arr(s, k) for k in ((dens,) if isinstance(dens, str) else dens))
        return n.sum() / max(d.sum(), 1e-9), n[idx].sum(axis=1) / np.maximum(d[idx].sum(axis=1), 1e-9)

    def cells(s, nums, dens):
        """[CI 字串, 與 base 的差]"""
        p, d = ratio(s, nums, dens)
        if base in M and s != base:
            pb, db = ratio(base, nums, dens)
            return [ci_str(p, d), diff_str(p - pb, d - db)]
        return [ci_str(p, d), "—"]

    def gap(s):
        """入聲字相似度 − 其他字相似度(越接近 0 越好)"""
        pe, de = ratio(s, "sim_et", "n_et")
        pn, dn = ratio(s, "sim_non", "n_non")
        return pe - pn, de - dn

    key_metrics = {                                   # 摘要表用:名稱 → 回傳 (點估計, draws) 的函式
        "一致度 CER(含聲調符號,越負越好)": lambda s: ratio(s, "ed_marks", "len_marks"),
        "入聲保留率(越正越好)": lambda s: ratio(s, "kept_et", "n_et"),
        "入聲−其他字 相似度差距(越正越好)": gap,
        "聲調記號一致率(越正越好)": lambda s: ratio(s, ("tone_et", "tone_non"), ("m_et", "m_non")),
    }
    n_et, n_non = arr(order[0], "n_et").sum(), arr(order[0], "n_non").sum()
    has_et = n_et >= 30

    # 0. 摘要
    if base in M and len(order) > 1:
        names = [k for k in key_metrics if has_et or "入聲" not in k]
        L += [f"## 摘要:各系統與 {base} 的差(百分點,95% CI)\n", "| 系統 | " + " | ".join(names) + " |", "|---|" + "---|" * len(names)]
        for s in order:
            if s == base:
                continue
            row = []
            for k in names:
                p, d = key_metrics[k](s)
                pb, db = key_metrics[k](base)
                row.append(diff_str(p - pb, d - db))
            L.append(f"| {s} | " + " | ".join(row) + " |")
        L.append("")

    # 1. 一致度 CER
    L += ["## 1. 與「真人錄音的辨識結果」的一致度(CER %,越低越好;只比拉丁字母)\n",
          f"| 系統 | 字母層級 CER | 與 {base} 的差 | 含聲調符號 CER | 與 {base} 的差 |", "|---|---|---|---|---|"]
    for s in order:
        L.append(f"| {s} | " + " | ".join(cells(s, "ed_letters", "len_letters") + cells(s, "ed_marks", "len_marks")) + " |")

    # 2. 入聲字 vs 其他字
    L += ["", "## 2. 入聲字 vs 其他字(入聲字 = 真人錄音的辨識結果裡,韻尾是 -p / -t / -k 的字)\n",
          f"真人辨識結果共 {n_et:.0f} 個入聲字、{n_non:.0f} 個其他字(多個訓練種子取平均)。"]
    if not has_et:
        L.append("入聲字太少,這一節略過。")
    else:
        L += ["\n**2a. 字的相似度**(合成語音的辨識結果與真人的逐字對齊後,同一個字的字母相似度 0–100;被聽漏的算 0;越高越好)\n",
              f"| 系統 | 入聲字 | 其他字 | 入聲字與 {base} 的差 | 入聲字 − 其他字 | 差距與 {base} 的差 |", "|---|---|---|---|---|---|"]
        for s in order:
            pe, de = ratio(s, "sim_et", "n_et")
            pn, dn = ratio(s, "sim_non", "n_non")
            dif, dd = "—", "—"
            if base in M and s != base:
                pb, db = ratio(base, "sim_et", "n_et")
                dif = diff_str(pe - pb, de - db)
                (gp, gd), (gbp, gbd) = gap(s), gap(base)
                dd = diff_str(gp - gbp, gd - gbd)
            L.append(f"| {s} | {ci_str(pe, de)} | {ci_str(pn, dn)} | {dif} | {diff_str(pe - pn, de - dn)} | {dd} |")
        L.append("\n「入聲字 − 其他字」為負:入聲字比其他字更常被聽走樣;越接近 0 表示入聲沒有特別吃虧。"
                 f"最後一欄是這個差距與 {base} 的差(正 = 入聲的劣勢比 {base} 小)。")
        L += ["\n**2b. 入聲有沒有被聽成入聲**(入聲保留率 = 真人聽成入聲的字,合成語音也聽成入聲〔韻尾 -p/-t/-k〕的比例,聽漏的算沒保留;"
              "其他字誤成入聲率 = 真人聽成其他字,合成語音卻聽出 -p/-t/-k 的比例)\n",
              f"| 系統 | 入聲保留率(越高越好) | 與 {base} 的差 | 其他字誤成入聲率(越低越好) | 與 {base} 的差 |", "|---|---|---|---|---|"]
        for s in order:
            L.append("| " + " | ".join([s] + cells(s, "kept_et", "n_et") + cells(s, "false_non", "m_non")) + " |")

    # 3. 聲調記號一致率
    L += ["", "## 3. 聲調記號一致率(%,越高越好)\n",
          f"對得上的字(字母相似度 ≥ {MATCH_SIM})之中,合成語音辨識出的聲調記號和真人的相同的比例(記號包含「無記號」)。\n",
          f"| 系統 | 全部 | 與 {base} 的差 | 入聲字 | 其他字 |", "|---|---|---|---|---|"]
    for s in order:
        L.append("| " + " | ".join([s] + cells(s, ("tone_et", "tone_non"), ("m_et", "m_non"))
                                   + [cells(s, "tone_et", "m_et")[0], cells(s, "tone_non", "m_non")[0]]) + " |")
    return "\n".join(L)


# ------------------------------------------------------------------ 輸入輸出
def read_tsv_rows(path):
    return list(csv.DictReader(open(path, encoding="utf-8"))) if Path(path).exists() else []


def write_rows(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["kind", "system", "k", "stem", "hyp"])
        w.writeheader()
        w.writerows(rows)


def parse_sys(specs):
    out = []
    for spec in specs:
        name, dirs = spec.split("=", 1)
        out.append((name, [Path(x).expanduser() for x in dirs.split(",") if x]))
    return out


def reference_syllables(dialect, stems):
    idx = {p.stem: p for p in C.textgrid_dir(dialect).rglob("*.TextGrid")}
    refs = {}
    for st in stems:
        if st not in idx:
            continue
        words = find_tier(read_textgrid(idx[st]), "words")
        syl = [t.strip() for _, _, t in words if t.strip().lower() not in C.MFA_SILENCE]
        if syl:
            refs[st] = syl
    return refs


def to16k(src, dst):
    if dst.exists():
        return
    import soundfile as sf
    x, sr = sf.read(str(src), dtype="float32", always_2d=True)
    x = x.mean(axis=1)
    if sr != 16000:
        try:
            import torch
            import torchaudio
            y = torchaudio.functional.resample(torch.from_numpy(x)[None], sr, 16000)[0].numpy()
        except Exception:  # noqa: BLE001
            from scipy.signal import resample_poly
            g = math.gcd(sr, 16000)
            y = resample_poly(x, 16000 // g, sr // g).astype("float32")
    else:
        y = x
    dst.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(dst), y, 16000, subtype="PCM_16")


def transcribe_omni(files, model, lang, batch, chunk=64):
    """lang = 'none':不給語言碼(LLM 模型會自己判斷;官方說這樣品質可能較差,只用來診斷語言設定有沒有影響)。"""
    from omnilingual_asr.models.inference.pipeline import ASRInferencePipeline
    pipe = ASRInferencePipeline(model_card=model)
    out = []
    for i in range(0, len(files), chunk):
        part = [str(f) for f in files[i:i + chunk]]
        res = pipe.transcribe(part, lang=None if lang.lower() in ("none", "") else [lang] * len(part), batch_size=batch)
        out += [r if isinstance(r, str) else str(r.get("text", r)) if isinstance(r, dict) else str(r) for r in res]
        print(f"  已辨識 {min(i + chunk, len(files))}/{len(files)}")
    return out


def transcribe_whisper(files, model, language, batch, chunk=64):
    import torch
    from transformers import pipeline
    cuda = torch.cuda.is_available()
    pipe = pipeline("automatic-speech-recognition", model=model, device=0 if cuda else -1,
                    torch_dtype=torch.float16 if cuda else torch.float32)
    gk = {"task": "transcribe"}
    if language:
        gk["language"] = language
    out = []
    for i in range(0, len(files), chunk):
        res = pipe([str(f) for f in files[i:i + chunk]], batch_size=batch, generate_kwargs=gk)
        out += [r["text"] for r in res]
        print(f"  已辨識 {min(i + chunk, len(files))}/{len(files)}")
    return out


def is_odd_char(c):
    """辨識結果裡不屬於白話字的字元:非拉丁字母的文字(漢字、韓文、泰文…)與數字。"""
    cat = unicodedata.category(c)
    if cat.startswith("L"):
        return not unicodedata.name(c, "").startswith("LATIN")
    return cat.startswith("N")


def show_examples(stems, ref_syl, real_hyp, syn_hyp, order, n=6):
    L = ["## 辨識輸出範例(ASR 用哪種拼寫、有沒有以音節分隔、有沒有聲調符號)\n"]
    for st in stems[:n]:
        L.append(f"- `{st}`\n  - 標準答案:{' '.join(ref_syl.get(st, []))}\n  - ASR(真人):{real_hyp.get(st, '')!r}")
        for s in order:
            h = syn_hyp.get((s, "0", st))
            if h is not None:
                L.append(f"  - ASR({s}):{h!r}")
    odd = sorted({c for h in real_hyp.values() for c in h if is_odd_char(c)})
    n_tok = [len(tokens(h)) for h in real_hyp.values()]
    n_ref = [len(ref_syl.get(st, [])) for st in real_hyp]
    share = np.mean([any(is_odd_char(c) for c in h) for h in real_hyp.values()])
    L += ["", f"每句平均詞數:ASR {np.mean(n_tok):.1f}、標準答案音節 {np.mean(n_ref):.1f}",
          f"真人辨識結果含非拉丁字母 / 數字(漢字、韓文、泰文、數字…,多半是辨識幻覺,指標會忽略這些字)的句子 {100 * share:.0f}%;"
          f"出現過的這類字元 {len(odd)} 種:{''.join(odd[:40])!r}{'…' if len(odd) > 40 else ''}\n"]
    return "\n".join(L)


def split_rows(rows):
    """辨識結果列 → (真人 {stem: 辨識結果}, 合成 {(系統, 種子, stem): 辨識結果}, 系統順序)。
    種子是字串:第一個生成種子 = '0'、'1'…(訓練種子序號);其他生成種子 = '0g2345'。"""
    real_hyp, syn_hyp, order = {}, {}, []
    for r in rows:
        if r["kind"] == "real":
            real_hyp[r["stem"]] = r["hyp"]
        else:
            syn_hyp[(r["system"], r["k"], r["stem"])] = r["hyp"]
            if r["system"] not in order:
                order.append(r["system"])
    return real_hyp, syn_hyp, order


def dialect_metrics(dialect, rows):
    """一個腔調的辨識結果 → (M, 系統順序, 真人辨識, 合成辨識, 標準答案音節, 因真人辨識不可靠而排除的句數)。"""
    real_hyp, syn_hyp, order = split_rows(rows)
    ref_syl = reference_syllables(dialect, list(real_hyp))
    ref_n = {st: len(v) for st, v in ref_syl.items()}
    M = per_stem_metrics(real_hyp, ref_n, syn_hyp)
    n_excl = sum(1 for st, h in real_hyp.items() if not reliable(h, ref_n.get(st, 0)))
    return M, order, real_hyp, syn_hyp, ref_syl, n_excl


def load_and_report(args, rows=None):
    path = Path(args.out_dir) / f"transcripts_{tag_of(args)}.csv"
    rows = rows if rows is not None else read_tsv_rows(path)
    if not rows:
        sys.exit(f"★ 找不到辨識結果 {path}:先跑 run")
    M, order, real_hyp, syn_hyp, ref_syl, n_excl = dialect_metrics(args.dialect, rows)
    base = args.base if args.base in M or not order else order[0]
    text = (f"# A1 語音辨識檢查:{args.dialect}({tag_of(args)})\n\n"
            + show_examples(sorted(real_hyp), ref_syl, real_hyp, syn_hyp, order) + "\n"
            + build_report(M, order, base, args.n_boot, n_excl))
    out = Path(args.out_dir) / f"a1_{tag_of(args)}_{args.dialect}.md"
    out.write_text(text, encoding="utf-8")
    print(text)
    print(f"\n→ {out}")


def cmd_pool(args):
    """把各腔調已辨識好的結果合併成一份報告(句子數加倍,信賴區間比較窄)。"""
    root = Path(args.out_root).expanduser()
    per, n_excl = {}, 0
    for d in args.dialects:
        path = root / d / f"transcripts_{tag_of(args)}.csv"
        rows = read_tsv_rows(path)
        if not rows:
            sys.exit(f"★ 找不到 {path}:先對 {d} 跑 run")
        M, order, _, _, _, ne = dialect_metrics(d, rows)
        per[d] = (M, order)
        n_excl += ne
    order = [s for s in per[args.dialects[0]][1] if all(s in M for M, _ in per.values())]
    pooled = {s: {} for s in order}
    for d, (M, _) in per.items():
        for s in order:
            for key, dct in M[s].items():
                pooled[s].setdefault(key, {}).update({f"{d}:{st}": v for st, v in dct.items()})
    base = args.base if args.base in pooled or not order else order[0]
    text = f"# A1 語音辨識檢查:合併 {' + '.join(args.dialects)}({tag_of(args)})\n\n" + build_report(pooled, order, base, args.n_boot, n_excl)
    out = root / f"a1_{tag_of(args)}_pooled.md"
    out.write_text(text, encoding="utf-8")
    print(text)
    print(f"\n→ {out}")


def tag_of(args):
    """辨識結果的檔名標記:後端 + 模型 + 語言設定;換任何一項都不會讀到舊的辨識結果。"""
    model = args.model or DEFAULT_MODEL[args.backend]
    lang = args.lang if args.backend == "omni" else (args.whisper_lang or "auto")
    return re.sub(r"[^A-Za-z0-9._-]+", "_", f"{args.backend}_{model}_{lang}")


def cmd_run(args):
    systems = parse_sys(args.sys)
    gens = [int(x) for x in args.gen_seeds.split(",") if x] if args.gen_seeds else [C.GEN_SEEDS[0]]
    stem_sets = [{p.name[:-len(f"_g{gs}.wav")] for p in (d / "wav").glob(f"*_g{gs}.wav")} for _, ds in systems for d in ds for gs in gens]
    stems = sorted(set.intersection(*stem_sets))
    if args.n_sent:
        stems = stems[:args.n_sent]
    real = {}
    for line in open(C.filelist("joint", args.dialect, "test"), encoding="utf-8"):
        if line.strip():
            wav = line.split("|")[0]
            real[Path(wav).stem] = wav
    stems = [s for s in stems if s in real]
    if not stems:
        sys.exit("★ 找不到可用的句子(合成音檔的 wav/ 與測試清單要有共同的檔名)")
    print(f"{len(stems)} 句;系統:{[n for n, _ in systems]};生成種子:{gens}")
    out_dir = Path(args.out_dir).expanduser()
    tpath = out_dir / f"transcripts_{tag_of(args)}.csv"
    rows = read_tsv_rows(tpath)
    done = {(r["kind"], r["system"], r["k"], r["stem"]) for r in rows}
    todo = []                                    # (kind, system, k, stem, 16k 路徑, 原始音檔)
    for st in stems:
        todo.append(("real", "real", "0", st, out_dir / "audio16k" / "real" / f"{st}.wav", Path(real[st])))
    for name, ds in systems:
        for k, d in enumerate(ds):
            for gs in gens:
                kk = str(k) if gs == C.GEN_SEEDS[0] else f"{k}g{gs}"          # 預設的第一個生成種子沿用原本的編號
                for st in stems:
                    todo.append(("syn", name, kk, st, out_dir / "audio16k" / f"{name}_{kk}" / f"{st}.wav", d / "wav" / f"{st}_g{gs}.wav"))
    todo = [t for t in todo if (t[0], t[1], t[2], t[3]) not in done]
    print(f"待辨識 {len(todo)} 個音檔(已完成 {len(rows)})")
    if todo:
        for t in todo:
            to16k(t[5], t[4])
        files = [t[4] for t in todo]
        if args.backend == "omni":
            hyps = transcribe_omni(files, args.model, args.lang, args.batch)
        else:
            hyps = transcribe_whisper(files, args.model, args.whisper_lang, args.batch)
        for t, h in zip(todo, hyps):
            rows.append({"kind": t[0], "system": t[1], "k": t[2], "stem": t[3], "hyp": h})
        write_rows(tpath, rows)
    load_and_report(args, rows)


DEFAULT_MODEL = {"omni": "omniASR_LLM_1B_v2", "whisper": "formospeech/whisper-large-v3-taiwanese-hakka"}


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("run", "analyze", "pool"):
        p = sub.add_parser(name)
        if name == "pool":
            p.add_argument("--out-root", required=True, help="各腔調的資料夾(<out-root>/<腔調>/)的上一層")
            p.add_argument("--dialects", nargs="+", default=list(C.DIALECTS), choices=C.DIALECTS)
        else:
            p.add_argument("--dialect", required=True, choices=C.DIALECTS)
            p.add_argument("--out-dir", required=True)
        p.add_argument("--backend", default="omni", choices=("omni", "whisper"))
        p.add_argument("--base", default="base", help="比較基準的系統名稱")
        p.add_argument("--n-boot", type=int, default=5000)
        p.add_argument("--model", help=f"omni:模型卡名稱(預設 {DEFAULT_MODEL['omni']});whisper:Hugging Face 模型 id(預設 {DEFAULT_MODEL['whisper']})")
        p.add_argument("--lang", default="hak_Latn", help="omni 的語言代碼;none = 不給語言碼")
        p.add_argument("--whisper-lang", default=None)
        if name == "run":
            p.add_argument("--sys", action="append", required=True, metavar="NAME=DIR1,DIR2,...",
                           help="D1 評估資料夾(要有 wav/);多個訓練種子用逗號分隔")
            p.add_argument("--batch", type=int, default=8)
            p.add_argument("--n-sent", type=int, default=0, help="只用前 N 句(試跑用)")
            p.add_argument("--gen-seeds", default=None, help="要辨識的生成種子,逗號分隔(預設只用第一個:1234);已辨識過的會跳過")
    args = ap.parse_args()
    args.model = args.model or DEFAULT_MODEL[args.backend]
    if args.cmd == "pool":
        cmd_pool(args)
        return
    args.out_dir = str(Path(args.out_dir).expanduser())
    if args.cmd == "run":
        cmd_run(args)
    else:
        load_and_report(args)


if __name__ == "__main__":
    main()
