"""R2:用微調好的 Raon-OpenTTS 合成評估句,輸出格式與 c_eval.py --synth 相同,之後直接接 D1(MFA 從波形量母音)。

句子選法與 c_eval.py 相同(每位語者依檔名排序後用種子 20260716 抽 --per-spk 句,× 5 個生成種子),
所以和 Matcha-TTS / VITS 比的是同一批句子、同一套量法。
語者:模型沒有語者編號,改用「同一位語者的一句訓練句」當聲音提示(r_prep.py 選好的,固定不變)。
整句總長(模型沒有時長預測器,必須由外部給):
  --total rate    依該語者訓練句的語速估(秒數 = a + b × 字元數)── 主要結果,完全不看測試句的真實長度
  --total oracle  直接用真人錄音的長度 ── 診斷用:總長給對時,模型內部分配每個音的長短準不準
輸出:<eval>/wav/<檔名>_g<種子>.wav(轉成 22.05 kHz,與其他系統相同)、bounds/<檔名>.json(沒有逐音素時長,
frames 為 null,d1_compare.py 會略過 attn 相關的比較)、mcd.csv、raon_info.json。
MCD 注意:模型與聲碼器都是 16 kHz,8 kHz 以上沒有能量;MCD 與 22.05 kHz 的系統不完全可比。

    python r_synth.py --exp raon_fair1h_sixian_s1234 --which step20000 --dialect sixian --name raon_sixian_step20000
"""
import argparse
import json
import sys
import time
from collections import Counter
from dataclasses import asdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from hakka_c import config as C  # noqa: E402
from hakka_c import raon_io as R  # noqa: E402
from hakka_c.structure import StructError, build_utt  # noqa: E402
from hakka_c.textgrid import read_textgrid  # noqa: E402


def synth_list(rows, per_spk, exclude):
    """與 c_eval.synth_list 相同。"""
    import random
    from collections import defaultdict
    by = defaultdict(list)
    for r in sorted(rows, key=lambda r: r["stem"]):
        if r["stem"] not in exclude:
            by[r["spk"]].append(r)
    rng = random.Random(C.SPLIT_SEED)
    out = []
    for spk in sorted(by):
        pool = by[spk]
        out += sorted(rng.sample(pool, min(per_spk, len(pool))), key=lambda r: r["stem"])
    return out


def symbol_names(pua_text, pua_map):
    """Matcha 清單的 PUA 文字 → 符號名稱(與 Engine.text 相同:先過 hakka_cleaners)。"""
    sys.path.insert(0, str(C.MATCHA))
    from matcha.text import text_to_sequence
    _, clean = text_to_sequence(pua_text, ["hakka_cleaners"])
    return [pua_map[ch] for ch in clean]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", required=True)
    ap.add_argument("--which", default="last")
    ap.add_argument("--dialect", required=True, choices=C.DIALECTS)
    ap.add_argument("--split", default="test", choices=("val", "test"))
    ap.add_argument("--total", default="rate", choices=("rate", "oracle"))
    ap.add_argument("--per-spk", type=int, default=50)
    ap.add_argument("--gen-seeds", default=",".join(map(str, C.GEN_SEEDS)))
    ap.add_argument("--exclude-stems")
    ap.add_argument("--no-mcd", action="store_true")
    ap.add_argument("--name", required=True)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    import soundfile as sf
    import torch
    import torchaudio.functional as AF
    from hakka_c.pua import resolve_pua_map

    ckpt = R.find_raon_ckpt(args.exp, args.which)
    out = C.OUT_ROOT / "eval" / args.name
    (out / "wav").mkdir(parents=True, exist_ok=True)
    (out / "bounds").mkdir(exist_ok=True)
    dd = R.data_dir(args.dialect)
    rows = R.load_jsonl(dd / f"{args.split}.jsonl")
    rate = {int(k): v for k, v in json.load(open(dd / "rate.json")).items()}
    prompts = {int(k): v for k, v in json.load(open(dd / "prompts.json", encoding="utf-8")).items()}
    pua_map, _ = resolve_pua_map(["hakka_cleaners"], verbose=False)
    pua_text = {}
    fl = C.filelist("joint", args.dialect, "test") if args.split == "test" else None
    if fl is None:
        import yaml
        v = Path(yaml.safe_load(open(C.MATCHA / "configs" / "data" / f"fair1h_joint_{args.dialect}.yaml",
                                     encoding="utf-8"))["valid_filelist_path"])
        fl = v if v.is_absolute() else C.MATCHA / v
    for line in open(fl, encoding="utf-8"):
        p = line.rstrip("\n").split("|")
        if len(p) >= 3:
            pua_text[Path(p[0]).stem] = p[2]

    # 和 c_eval.py 一樣,只留下結構對得上真人 TextGrid 的句子
    tgs = {p.stem: p for p in C.textgrid_dir(args.dialect).rglob("*.TextGrid")}
    ok, why = [], Counter()
    for r in rows:
        try:
            names = symbol_names(pua_text[r["stem"]], pua_map)
            utt = build_utt(r["stem"], names, read_textgrid(tgs[r["stem"]]))
        except (StructError, KeyError) as e:
            why[str(e)[:50]] += 1
            continue
        ok.append({**r, "names": names, "syllables": [asdict(s) for s in utt.syllables]})
    exclude = set(l.strip() for l in open(args.exclude_stems, encoding="utf-8")) if args.exclude_stems else set()
    # 與 c_eval.py 相同:先從整份清單抽句,再留下結構對得上的(所以和其他系統是同一批句子)
    spk_of = {}
    for line in open(fl, encoding="utf-8"):
        p = line.rstrip("\n").split("|")
        if len(p) >= 3:
            spk_of[Path(p[0]).stem] = int(p[1])
    by_stem = {r["stem"]: r for r in ok}
    todo = [by_stem[r["stem"]] for r in synth_list([{"stem": k, "spk": v} for k, v in spk_of.items()], args.per_spk, exclude)
            if r["stem"] in by_stem]
    seeds = [int(s) for s in args.gen_seeds.split(",")]
    print(f"checkpoint {ckpt}\n句子 {len(todo)} × 生成種子 {len(seeds)};總長 {args.total};略過 {dict(why) or '無'}\n輸出 {out}")

    model, _, step = R.load_finetuned(ckpt, args.device)
    voc = R.load_vocoder(args.device)
    ev = None
    if not args.no_mcd:
        sys.path.insert(0, str(C.ANALYSIS))
        import evaluate_v2 as ev
    prompt_audio = {s: sf.read(p["wav16"], dtype="float32")[0] for s, p in prompts.items()}
    mcd_rows, info, t0 = [], [], time.time()
    for j, r in enumerate(todo):
        if args.total == "oracle":
            gen = R.mel_frames(round(r["sec"] * R.SR16))
        else:
            gen = round(R.rate_seconds(rate, r["spk"], r["text"]) * R.FPS16)
        json.dump({"stem": r["stem"], "names": r["names"], "frames": None, "syllables": r["syllables"],
                   "params": {"arch": "raon", "total": args.total, "gen_frames": gen}},
                  open(out / "bounds" / f"{r['stem']}.json", "w"), ensure_ascii=False)
        ref = None
        if ev is not None:
            a, sr = sf.read(r["wav"], dtype="float64", always_2d=True)
            assert sr == C.SR, (r["wav"], sr)
            f0_ref, mc_ref = ev.analyze_wav(a.mean(1))
            ref = mc_ref
        p = prompts[r["spk"]]
        for gs in seeds:
            w16 = R.synthesize(model, voc, prompt_audio[r["spk"]], p["text"], r["text"], gen, gs, args.device)
            syn = AF.resample(torch.from_numpy(w16), R.SR16, C.SR).numpy().astype(np.float64)
            sf.write(out / "wav" / f"{r['stem']}_g{gs}.wav", syn, C.SR)
            info.append({"stem": r["stem"], "gen_seed": gs, "gen_frames": gen, "syn_sec": round(len(syn) / C.SR, 4),
                         "ref_sec": r["sec"]})
            if ev is not None:
                _, mc_syn = ev.analyze_wav(syn)
                mcd_rows.append({"stem": r["stem"], "spk": r["spk"], "gender": C.SPK_GENDER.get(r["spk"], "?"),
                                 "gen_seed": gs, "mcd": float(ev.mcd_from_mcep(mc_syn, ref))})
        if (j + 1) % 10 == 0:
            print(f"  合成 {j + 1}/{len(todo)}  {time.time() - t0:.0f}s", flush=True)
    if mcd_rows:
        from hakka_c.csvio import write_rows
        write_rows(out / "mcd.csv", mcd_rows)
        m = np.array([x["mcd"] for x in mcd_rows])
        print(f"MCD 平均 {m.mean():.3f} dB(16 kHz 模型,與 22.05 kHz 系統不完全可比)")
    rel = np.array([x["syn_sec"] / x["ref_sec"] for x in info])
    json.dump({"exp": args.exp, "ckpt": str(ckpt), "step": step, "total": args.total, "split": args.split,
               "n_utts": len(todo), "seeds": seeds, "len_ratio_mean": float(rel.mean()),
               "len_ratio_abs_err_pct": float(100 * np.abs(rel - 1).mean()), "rows": info},
              open(out / "raon_info.json", "w"), indent=1, ensure_ascii=False)
    print(f"合成長度 / 真人長度:平均 {rel.mean():.3f}(平均絕對差 {100 * np.abs(rel - 1).mean():.1f}%)\n→ {out}")


if __name__ == "__main__":
    main()
