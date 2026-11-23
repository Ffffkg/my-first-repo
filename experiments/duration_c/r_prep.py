"""R0:準備 Raon-OpenTTS 微調與評估用的資料。不需要 GPU,幾分鐘。需要 raon 環境(或 tts 環境 + torchaudio)。

和 Matcha-TTS / VITS 用完全相同的句子:
  train / val = configs/data/fair1h_joint_<腔>.yaml 的訓練、驗證清單(Base 用的 1 小時清單)
  test        = data/hakka/joint/<腔>/test.txt
模型輸入文字:真人錄音 TextGrid words 層的拼音音節(與 D1 量測時給 MFA 的 .lab 相同),音節之間一個空白。
  官方模型是英文字元級輸入,客語拼音大多是拉丁字母與數字,可以沿用預訓練的字元嵌入;缺少的字元才新增。
音檔:轉成 16 kHz(官方模型的取樣率)存到 ~/hakka_tts/raon/wav16k/<腔>/。

輸出(~/hakka_tts/raon/):
  data/<腔>/{train,val,test}.jsonl   每句 {stem, spk, wav, wav16, text, sec}
  data/<腔>/rate.json                每位語者「秒數 = a + b × 字元數」(推論時估整句總長)
  data/<腔>/prompts.json             每位語者的聲音提示句(訓練句中 3–6 秒、最接近 4.5 秒者)
  vocab_hakka.txt                    官方詞表 + 新增字元(兩腔共用)

    python r_prep.py                 # 兩腔
"""
import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from hakka_c import config as C  # noqa: E402
from hakka_c import raon_io as R  # noqa: E402
from hakka_c.textgrid import find_tier, read_textgrid  # noqa: E402


def read_filelist(path):
    rows = []
    for line in open(path, encoding="utf-8"):
        p = line.rstrip("\n").split("|")
        if len(p) >= 3:
            rows.append({"wav": p[0], "spk": int(p[1]), "pua": p[2]})
    return rows


def yaml_list(d, key):
    import yaml
    v = Path(yaml.safe_load(open(C.MATCHA / "configs" / "data" / f"fair1h_joint_{d}.yaml", encoding="utf-8"))[key])
    return v if v.is_absolute() else C.MATCHA / v


def to16k(src, dst):
    import soundfile as sf
    import torch
    import torchaudio.functional as AF
    a, sr = sf.read(str(src), dtype="float32", always_2d=True)
    a = torch.from_numpy(a.mean(1))
    if sr != R.SR16:
        a = AF.resample(a, sr, R.SR16)
    dst.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(dst), a.numpy(), R.SR16, subtype="PCM_16")
    return len(a) / R.SR16


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dialects", nargs="*", default=list(C.DIALECTS))
    args = ap.parse_args()
    import soundfile as sf
    texts, report = [], []
    for d in args.dialects:
        tgs = {p.stem: p for p in C.textgrid_dir(d).rglob("*.TextGrid")}
        out = R.data_dir(d)
        out.mkdir(parents=True, exist_ok=True)
        lists = {"train": yaml_list(d, "train_filelist_path"), "val": yaml_list(d, "valid_filelist_path"),
                 "test": C.filelist("joint", d, "test")}
        split_rows = {}
        for split, fl in lists.items():
            rows, why = [], Counter()
            for r in read_filelist(fl):
                wav = Path(r["wav"]) if Path(r["wav"]).is_absolute() else C.MATCHA / r["wav"]
                stem = wav.stem
                if stem not in tgs:
                    why["沒有 TextGrid"] += 1
                    continue
                words = [t for _, _, t in find_tier(read_textgrid(tgs[stem]), "words")
                         if t.strip().lower() not in C.MFA_SILENCE]
                if not words:
                    why["TextGrid 沒有詞"] += 1
                    continue
                w16 = R.RAON / "wav16k" / d / f"{stem}.wav"
                sec = sf.info(str(w16)).duration if w16.exists() else to16k(wav, w16)
                rows.append({"stem": stem, "spk": r["spk"], "wav": str(wav), "wav16": str(w16),
                             "text": R.words_text(words), "sec": round(sec, 4)})
            with open(out / f"{split}.jsonl", "w", encoding="utf-8") as f:
                for r in rows:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
            split_rows[split] = rows
            texts += [r["text"] for r in rows]
            report.append(f"{d} {split}:{len(rows)} 句({sum(r['sec'] for r in rows) / 3600:.2f} 小時)"
                          f"{';略過 ' + str(dict(why)) if why else ''}")
        rate = R.fit_rate(split_rows["train"])
        json.dump({str(k): v for k, v in rate.items()}, open(out / "rate.json", "w"), indent=2)
        prompts = R.pick_prompt(split_rows["train"])
        missing = sorted({r["spk"] for r in split_rows["test"]} - set(prompts))
        if missing:
            sys.exit(f"★ {d}:語者 {missing} 沒有 3–6 秒的訓練句可當聲音提示")
        json.dump({str(k): v for k, v in prompts.items()}, open(out / "prompts.json", "w"), indent=2, ensure_ascii=False)
        for spk, (a, b) in rate.items():
            report.append(f"  語者 {spk}:秒數 ≈ {a:.2f} + {b:.4f} × 字元數;提示句 {prompts[spk]['stem']}({prompts[spk]['sec']:.2f} 秒)")
        test_err = [abs(R.rate_seconds(rate, r["spk"], r["text"]) - r["sec"]) / r["sec"] for r in split_rows["test"]]
        report.append(f"  用語速估測試句總長:平均相對誤差 {100 * sum(test_err) / max(len(test_err), 1):.1f}%")

    # 詞表(兩腔共用;已存在就只檢查沒有新字元)
    _, vocab_src = R.find_pretrained()
    old = R.read_vocab(vocab_src)
    vpath = R.RAON / "vocab_hakka.txt"
    if vpath.exists():
        cur = R.read_vocab(vpath)
        extra = sorted({ch for t in texts for ch in t} - set(cur))
        if extra:
            sys.exit(f"★ {vpath} 已存在但缺少字元 {extra}:刪掉它、兩腔一起重跑 r_prep.py(已訓練的模型要重訓)")
        report.append(f"詞表 {vpath}:{len(cur)} 個字元(沿用)")
    else:
        m, new = R.extend_vocab(old, texts)
        R.write_vocab(m, vpath)
        assert R.read_vocab(vpath) == m
        report.append(f"詞表 {vpath}:官方 {len(old)} 個字元,新增 {len(new)} 個 {new}")
    json.dump({"official_vocab": str(vocab_src), "n_official": len(old)}, open(R.RAON / "vocab_hakka.json", "w"), indent=2)
    print("\n".join(report))


if __name__ == "__main__":
    main()
