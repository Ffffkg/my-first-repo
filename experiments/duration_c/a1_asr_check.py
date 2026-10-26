"""A1:用語音辨識(ASR)模型當「獨立於 MFA」的客觀檢查——合成語音能不能被辨識得跟真人一樣,入聲字有沒有特別吃虧。

為什麼:我們所有時長指標都靠 MFA 量。這裡改用一個完全不同的系統(Meta Omnilingual ASR,有 hak_Latn 客語;
也可選 Whisper 客語微調)去「聽」合成語音,看辨識結果跟聽真人錄音的結果差多少。

三個指標(都是越低 / 越高越好,見報告內說明):
  1. 一致度 CER:ASR(合成)對 ASR(同一句真人錄音)的字元錯誤率。不需要知道 ASR 用哪種拼寫,兩邊都是它自己的輸出。
       字母層級 = 忽略聲調符號與數字;含聲調符號 = 聲調符號、數字也算。
  2. 入聲字一致度:用標準答案(TextGrid 的客語拼音音節)決定哪些位置是入聲(-b/-d/-g),
       再看 ASR(真人)在那些位置的詞,被 ASR(合成)保留了多少(0–1)。需要 ASR 輸出以音節/詞分隔,
       且詞數與標準答案相同的句子(報告會列出可用句數);不滿足就只給 1.、3.。
  3. 對標準答案的 CER(僅供參考):若 ASR 的拼寫與客語拼音不同,這個數字會被拼寫差異墊高,
       所以一定要看「真人錄音」那一列當底線。

    python a1_asr_check.py run --dialect sixian --backend omni --model omniASR_LLM_1B_v2 \\
        --sys base=$E/base_sixian_step40000,... --sys c6=... --out-dir ~/hakka_tts/exp_c/a1/sixian
    python a1_asr_check.py analyze --dialect sixian --out-dir ~/hakka_tts/exp_c/a1/sixian   # 只重算報告(模型、語言設定要與 run 相同)

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
from hakka_c.structure import is_entering_syllable  # noqa: E402
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


def cer(hyp, ref):
    return lev(hyp, ref) / max(len(ref), 1)


def ref_letters(refs):
    """標準答案音節(如 sad5 ngin2)→ 去掉聲調數字的字母串。"""
    return "".join(re.sub(r"\d+", "", s) for s in refs).lower()


def align_tokens(ref, hyp):
    """ref、hyp:已正規化的 token 列。回傳每個 ref token 的相似度(0–1,被刪掉 = 0)。"""
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
    sims = [0.0] * n
    i, j = n, m
    while i > 0 or j > 0:
        b = B[i][j]
        if b == 1:
            sims[i - 1] = 1.0 - sub_cost[i, j]
            i, j = i - 1, j - 1
        elif b == 2:
            i -= 1
        else:
            j -= 1
    return sims


# ------------------------------------------------------------------ 逐句指標與報告
def per_stem_metrics(real_hyp, ref_syl, syn_hyp):
    """syn_hyp: {(sys, k, stem): str}。回傳 {sys: {metric: {stem: 各種子平均}}}。"""
    acc = {}
    for (sysname, k, stem), hyp in syn_hyp.items():
        real = real_hyp.get(stem, "")
        refs = ref_syl.get(stem)
        if not real.strip() or not refs:
            continue
        m = {"cer_letters": cer(letters(hyp), letters(real)), "cer_marks": cer(with_marks(hyp), with_marks(real)),
             "ref_cer": cer(letters(hyp), ref_letters(refs))}
        rt = [t for t in (letters(t) for t in tokens(real)) if t]
        if len(rt) == len(refs):
            ht = [t for t in (letters(t) for t in tokens(hyp)) if t]
            sims = align_tokens(rt, ht)
            et = [is_entering_syllable(s) is True for s in refs]
            m.update(num_et=sum(s for s, e in zip(sims, et) if e), n_et=sum(et),
                     num_non=sum(s for s, e in zip(sims, et) if not e), n_non=len(et) - sum(et))
        for key, v in m.items():
            acc.setdefault(sysname, {}).setdefault(key, {}).setdefault(stem, []).append(v)
    return {s: {key: {st: float(np.mean(v)) for st, v in d.items()} for key, d in ms.items()} for s, ms in acc.items()}


def ci_str(point, draws, scale=100.0, digits=1):
    lo, hi = np.percentile(draws, [2.5, 97.5])
    return f"{scale * point:.{digits}f} [{scale * lo:.{digits}f}, {scale * hi:.{digits}f}]"


def diff_str(point, draws, scale=100.0, digits=1):
    lo, hi = np.percentile(draws, [2.5, 97.5])
    return f"{scale * point:+.{digits}f} [{scale * lo:+.{digits}f}, {scale * hi:+.{digits}f}]"


def build_report(M, order, base, real_floor, n_boot=5000):
    """M: per_stem_metrics 的結果。order:系統顯示順序;base:比較基準的系統名稱。"""
    L = []
    order = [s for s in order if s in M]
    if not order:
        return "(沒有任何可用的辨識結果:真人或合成的輸出是空的,或找不到標準答案。)"
    common = sorted(set.intersection(*[set(M[s]["cer_letters"]) for s in order]))
    if not common:
        return "(各系統沒有共同的句子可比較。)"
    L += [f"共同句數 {len(common)} 句。CI = 以句子重抽 {n_boot} 次的 95% 區間;「與 {base} 的差」為同一批句子的配對差。\n"]
    idx = boot_idx(len(common), n_boot)

    def arr(s, key, stems):
        return np.array([M[s][key][t] for t in stems])

    # 1. 一致度 CER
    L += ["## 1. 與「真人錄音的辨識結果」的一致度(CER %,越低越好)\n",
          f"| 系統 | 字母層級 CER | 與 {base} 的差 | 含聲調符號 CER | 與 {base} 的差 |", "|---|---|---|---|---|"]
    for s in order:
        cells = []
        for key in ("cer_letters", "cer_marks"):
            x = arr(s, key, common)
            cells.append(ci_str(x.mean(), x[idx].mean(axis=1)))
            if base in M and s != base:
                d = x - arr(base, key, common)
                cells.append(diff_str(d.mean(), d[idx].mean(axis=1)))
            else:
                cells.append("—")
        L.append(f"| {s} | {cells[0]} | {cells[1]} | {cells[2]} | {cells[3]} |")

    # 2. 入聲字一致度
    el = sorted(set.intersection(*[set(M[s].get("n_et", {})) for s in order]))
    L += ["", "## 2. 入聲字 vs 非入聲字的一致度(0–100,越高越好)\n"]
    if len(el) < 20:
        L.append(f"(可用句數只有 {len(el)} 句:ASR 輸出的詞數與標準答案的音節數不一致的句子太多,這一項略過。)")
    else:
        idx2 = boot_idx(len(el), n_boot, seed=43)
        L += [f"可用句數 {len(el)}(ASR 真人輸出的詞數 = 標準答案音節數)。",
              f"\n| 系統 | 入聲字 | 非入聲字 | 入聲字與 {base} 的差 | 入聲字 − 非入聲字 |", "|---|---|---|---|---|"]
        a = {s: {k: arr(s, k, el) for k in ("num_et", "n_et", "num_non", "n_non")} for s in order}
        for s in order:
            x = a[s]
            et = x["num_et"][idx2].sum(axis=1) / np.maximum(x["n_et"][idx2].sum(axis=1), 1)
            non = x["num_non"][idx2].sum(axis=1) / np.maximum(x["n_non"][idx2].sum(axis=1), 1)
            p_et, p_non = x["num_et"].sum() / max(x["n_et"].sum(), 1), x["num_non"].sum() / max(x["n_non"].sum(), 1)
            if base in a and s != base:
                b = a[base]
                bet = b["num_et"][idx2].sum(axis=1) / np.maximum(b["n_et"][idx2].sum(axis=1), 1)
                dif = diff_str(p_et - b["num_et"].sum() / max(b["n_et"].sum(), 1), et - bet)
            else:
                dif = "—"
            L.append(f"| {s} | {ci_str(p_et, et)} | {ci_str(p_non, non)} | {dif} | {diff_str(p_et - p_non, et - non)} |")
        L.append("\n「入聲字 − 非入聲字」為負:入聲字比非入聲字更常被辨識錯;越接近 0 表示入聲沒有特別吃虧。")

    # 3. 對標準答案的 CER
    L += ["", "## 3. 對標準答案(客語拼音)的 CER(%,僅供參考:ASR 拼寫與客語拼音不同時會被墊高)\n",
          "| 系統 | 字母層級 CER |", "|---|---|"]
    if real_floor is not None:
        L.append(f"| **真人錄音(底線)** | {100 * real_floor:.1f} |")
    for s in order:
        x = arr(s, "ref_cer", common)
        L.append(f"| {s} | {ci_str(x.mean(), x[idx].mean(axis=1))} |")
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


def show_examples(stems, ref_syl, real_hyp, syn_hyp, order, n=6):
    L = ["## 辨識輸出範例(先看這個:ASR 用哪種拼寫、有沒有以音節分隔、有沒有聲調符號)\n"]
    for st in stems[:n]:
        L.append(f"- `{st}`\n  - 標準答案:{' '.join(ref_syl.get(st, []))}\n  - ASR(真人):{real_hyp.get(st, '')!r}")
        for s in order:
            h = syn_hyp.get((s, 0, st))
            if h is not None:
                L.append(f"  - ASR({s}):{h!r}")
    chars = "".join(sorted({c for h in real_hyp.values() for c in h}))
    n_tok = [len(tokens(h)) for h in real_hyp.values()]
    n_ref = [len(ref_syl.get(st, [])) for st in real_hyp]
    same = [a == b for a, b in zip(n_tok, n_ref) if b]
    L += ["", f"真人輸出用到的字元:{chars!r}"]
    if same:
        L.append(f"每句平均詞數:ASR {np.mean(n_tok):.1f}、標準答案音節 {np.mean(n_ref):.1f};"
                 f"詞數與音節數相同的句子 {100 * np.mean(same):.0f}%\n")
    return "\n".join(L)


def load_and_report(args, rows=None):
    rows = rows if rows is not None else read_tsv_rows(Path(args.out_dir) / f"transcripts_{tag_of(args)}.csv")
    if not rows:
        sys.exit(f"★ 找不到辨識結果 {Path(args.out_dir) / f'transcripts_{tag_of(args)}.csv'}:先跑 run")
    real_hyp = {r["stem"]: r["hyp"] for r in rows if r["kind"] == "real"}
    syn_hyp, order = {}, []
    for r in rows:
        if r["kind"] == "syn":
            syn_hyp[(r["system"], int(r["k"]), r["stem"])] = r["hyp"]
            if r["system"] not in order:
                order.append(r["system"])
    ref_syl = reference_syllables(args.dialect, list(real_hyp))
    M = per_stem_metrics(real_hyp, ref_syl, syn_hyp)
    base = args.base if args.base in M or not order else order[0]
    rs = [cer(letters(h), ref_letters(ref_syl[st])) for st, h in real_hyp.items() if st in ref_syl and h.strip()]
    floor = float(np.mean(rs)) if rs else None
    text = f"# A1 語音辨識檢查:{args.dialect}({tag_of(args)})\n\n" + (
        show_examples(sorted(real_hyp), ref_syl, real_hyp, syn_hyp, order) + "\n"
        + build_report(M, order, base, floor, args.n_boot))
    out = Path(args.out_dir) / f"a1_{tag_of(args)}_{args.dialect}.md"
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
    gs = args.gen_seed or C.GEN_SEEDS[0]
    suffix = f"_g{gs}.wav"
    stem_sets = [{p.name[:-len(suffix)] for p in (d / "wav").glob(f"*{suffix}")} for _, ds in systems for d in ds]
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
    print(f"{len(stems)} 句;系統:{[n for n, _ in systems]}")
    out_dir = Path(args.out_dir).expanduser()
    tpath = out_dir / f"transcripts_{tag_of(args)}.csv"
    rows = read_tsv_rows(tpath)
    done = {(r["kind"], r["system"], r["k"], r["stem"]) for r in rows}
    todo = []                                    # (kind, system, k, stem, 16k 路徑)
    for st in stems:
        todo.append(("real", "real", "0", st, out_dir / "audio16k" / "real" / f"{st}.wav", Path(real[st])))
    for name, ds in systems:
        for k, d in enumerate(ds):
            for st in stems:
                todo.append(("syn", name, str(k), st, out_dir / "audio16k" / f"{name}_{k}" / f"{st}.wav", d / "wav" / f"{st}{suffix}"))
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
    for name in ("run", "analyze"):
        p = sub.add_parser(name)
        p.add_argument("--dialect", required=True, choices=C.DIALECTS)
        p.add_argument("--backend", default="omni", choices=("omni", "whisper"))
        p.add_argument("--out-dir", required=True)
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
            p.add_argument("--gen-seed", type=int, default=0)
    args = ap.parse_args()
    args.out_dir = str(Path(args.out_dir).expanduser())
    args.model = args.model or DEFAULT_MODEL[args.backend]
    if args.cmd == "run":
        cmd_run(args)
    else:
        load_and_report(args)


if __name__ == "__main__":
    main()
