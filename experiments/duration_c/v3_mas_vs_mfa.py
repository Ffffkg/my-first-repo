"""V3b:Base 模型「自己找的對齊」(MAS)和 MFA 時長差多少?Matcha-TTS 與 VITS、四縣與海陸並列。需要 GPU(幾分鐘)。

假設:+AD 在 VITS 海陸變差,是因為 VITS 在海陸「想要」的時長切法和 MFA 差很多,硬套 MFA 就互相衝突。
做法:用訓練好的 Base 模型(三個種子),在 1 小時訓練句上把真人錄音送進模型,讓它用 MAS 找出每個符號的音框數
(Matcha-TTS 用編碼器輸出對梅爾頻譜;VITS 用後驗編碼器的平均值經 flow 後的潛在表示),再和 MFA 時長比較:
  - 母音核長度(blank 依 half 規則分配)的平均絕對差、ρ、相差 ≤ 30 ms 的比例
  - 音節總長(音節內符號 + 音節內的 blank)的平均絕對差、ρ
  - 有多少音框被分給 blank(兩種對齊方式的慣例差異)
音節範圍取自 S3/S4 的音節分組表(由拼音文字決定)。

    python v3_mas_vs_mfa.py                       # 兩腔、三個種子
"""
import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from hakka_c import config as C  # noqa: E402
from hakka_c.durations import merge_blank  # noqa: E402
from hakka_c.stats import pearson  # noqa: E402
from s3_syllable_check import load_symbols  # noqa: E402

PATCH = Path(__file__).resolve().parent.parent / "matcha_patch" / "matcha" / "models" / "components" / "hakka_syllables.py"
SEEDS = (1234, 123, 2026)


def hs_module():
    spec = importlib.util.spec_from_file_location("hakka_syllables", PATCH)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def train_list(d):
    import yaml
    v = Path(yaml.safe_load(open(C.MATCHA / "configs" / "data" / f"fair1h_joint_{d}_mfadur.yaml", encoding="utf-8"))["train_filelist_path"])
    return v if v.is_absolute() else C.MATCHA / v


def vits_mas(model, x, spk, wav, T, device):
    import torch
    from hakka_c import vits_model as V
    a = torch.from_numpy(wav[: T * C.HOP]).float()[None].to(device)
    spec = V.spectrogram(a)
    sl = torch.tensor([T], device=device)
    xl = torch.tensor([x.shape[1]], device=device)
    sid = torch.tensor([spk], device=device)
    with torch.no_grad():
        _, m_p, logs_p, x_mask = model.enc_p(x, xl)
        g = model.emb_g(sid).unsqueeze(-1)
        _, m_q, _, y_mask = model.enc_q(spec, sl, g=g)
        z_p = model.flow(m_q * y_mask, y_mask, g=g)
        s = torch.exp(-2 * logs_p)
        nc = (torch.sum(-0.5 * np.log(2 * np.pi) - logs_p, [1], keepdim=True)
              + torch.matmul(-0.5 * (z_p ** 2).transpose(1, 2), s)
              + torch.matmul(z_p.transpose(1, 2), m_p * s)
              + torch.sum(-0.5 * (m_p ** 2) * s, [1], keepdim=True))
        mask = torch.unsqueeze(x_mask, 2) * torch.unsqueeze(y_mask, -1)
        attn = V._mas(nc.float(), mask.squeeze(1).float())             # (1, T, t_x)
    return attn[0].sum(0).round().long().cpu().numpy()


def compare(frames_a, frames_b, ids, groups, names, hs):
    """回傳逐音節 (入聲?, 母音 A, 母音 B, 音節 A, 音節 B)(ms),以及 blank 音框比例 (A, B)。"""
    syl, _ = hs.segment_from_groups(ids, groups)
    ma, mb = merge_blank(frames_a, "half") * C.HOP_MS, merge_blank(frames_b, "half") * C.HOP_MS
    nb = [p for p, i in enumerate(ids) if i != 0]           # 非 blank 位置 → 符號序號
    rows = []
    for k in range(max(groups) + 1 if groups else 0):
        pos = [p for p, s in enumerate(syl) if s == k]
        sym = [j for j, p in enumerate(nb) if syl[p] == k]
        anc = [j for j in sym if hs.classify(names[ids[nb[j]]])[0] == "anchor"]
        if len(anc) != 1:
            continue
        j = anc[0]
        ent = hs.classify(names[ids[nb[j]]])[1]
        rows.append((ent, ma[j], mb[j], frames_a[pos].sum() * C.HOP_MS, frames_b[pos].sum() * C.HOP_MS))
    blank = (frames_a[0::2].sum() / max(frames_a.sum(), 1), frames_b[0::2].sum() / max(frames_b.sum(), 1))
    return rows, blank


def summarize(rows, blanks):
    r = np.array([x[1:] for x in rows], float)
    et = np.array([x[0] for x in rows], bool)
    out = {}
    for lab, m in (("入聲", et), ("舒聲", ~et)):
        va, vb, sa, sb = r[m, 0], r[m, 1], r[m, 2], r[m, 3]
        out[lab] = (np.abs(vb - va).mean(), pearson(va, vb), 100 * (np.abs(vb - va) <= 30).mean(), (vb - va).mean(),
                    np.abs(sb - sa).mean(), pearson(sa, sb))
    b = np.array(blanks)
    return out, b[:, 0].mean(), b[:, 1].mean()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dialects", nargs="*", default=list(C.DIALECTS))
    ap.add_argument("--max-utts", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    import soundfile as sf
    import torch
    from hakka_c.matcha_io import Engine, find_ckpt, read_filelist
    from hakka_c.vits_io import VitsEngine, find_vits_ckpt
    hs = hs_module()
    names = [None] + list(load_symbols(C.MATCHA / "matcha" / "text" / "symbols.py"))

    L = ["# V3b:Base 模型自己找的對齊(MAS)vs MFA 時長(1 小時訓練句,三個種子平均)\n",
         "| 腔調 | 架構 | 類別 | 母音平均絕對差 (ms) | 母音 ρ | 母音相差 ≤ 30 ms | 母音平均差 MAS − MFA (ms) | 音節總長平均絕對差 (ms) | 音節總長 ρ |",
         "|---|---|---|---|---|---|---|---|---|"]
    blank_rows = []
    for d in args.dialects:
        table = json.load(open(C.OUT_ROOT / "s4" / f"syl_table_{d}.json", encoding="utf-8"))
        rows_in = read_filelist(train_list(d))
        if args.max_utts:
            rows_in = rows_in[: args.max_utts]
        for arch in ("Matcha-TTS", "VITS"):
            per_seed = []
            for s in SEEDS:
                if arch == "Matcha-TTS":
                    eng = Engine(find_ckpt(f"fair1h_joint_{d}_scratch_s{s}", "step40000"), args.device)
                else:
                    eng = VitsEngine(find_vits_ckpt(f"vits_base_fair1h_{d}_s{s}", "step200000"), args.device)
                rows, blanks = [], []
                for r in rows_in:
                    wav = Path(r["wav"]) if Path(r["wav"]).is_absolute() else C.MATCHA / r["wav"]
                    groups = table.get(r["text"])
                    dur = wav.parent.parent / "durations" / f"{wav.stem}.npy"
                    if groups is None or not dur.exists():
                        continue
                    mfa = np.load(dur).astype(float)
                    x, _ = eng.text(r["text"])
                    ids = x[0].tolist()
                    T = int(mfa.sum())
                    if arch == "Matcha-TTS":
                        mu_x, x_mask, _, _ = eng.encode(x, r["spk"])
                        y = eng.mel(wav)[:, :, :T]
                        if y.shape[-1] != T:
                            continue
                        mas = eng.mas(mu_x, x_mask, y).astype(float)
                    else:
                        a, _ = sf.read(str(wav), dtype="float32", always_2d=True)
                        if len(a) < T * C.HOP:
                            continue
                        mas = vits_mas(eng.model, x, r["spk"], a.mean(1), T, eng.device).astype(float)
                    if len(mas) != len(mfa):
                        continue
                    rr, bb = compare(mfa, mas, ids, groups, names, hs)
                    rows += rr
                    blanks.append(bb)
                per_seed.append(summarize(rows, blanks))
                del eng
                torch.cuda.empty_cache()
            for lab in ("入聲", "舒聲"):
                m = np.mean([p[0][lab] for p in per_seed], axis=0)
                L.append(f"| {d} | {arch} | {lab} | {m[0]:.1f} | {m[1]:.3f} | {m[2]:.1f}% | {m[3]:+.1f} | {m[4]:.1f} | {m[5]:.3f} |")
            blank_rows.append(f"| {d} | {arch} | {100 * np.mean([p[1] for p in per_seed]):.1f}% | "
                              f"{100 * np.mean([p[2] for p in per_seed]):.1f}% |")
            print(f"  完成 {d} {arch}", file=sys.stderr)
    L += ["", "## 分給 blank 的音框比例\n", "| 腔調 | 架構 | MFA 時長 | MAS |", "|---|---|---|---|"] + blank_rows
    L += ["", "判讀:若 VITS 的 MAS 在海陸和 MFA 差得特別多(四縣、Matcha-TTS 較接近),代表 VITS 在海陸自己學到的時長切法"
          "和 MFA 不同,強制套用 MFA 時長會和它的聲學表示衝突。"]
    print("\n".join(L))


if __name__ == "__main__":
    main()
