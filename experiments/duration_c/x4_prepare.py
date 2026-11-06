"""X4 第 1 步:建立「只用 1 h 目標腔資料訓練 MFA」所需的獨立資料樹。不需要 GPU。

主實驗的 MFA 用整個腔調的錄音訓練;X4 改成只用 1 h 訓練子集訓練 MFA,再用它產生時長標籤。
為了不覆蓋主實驗的 durations/*.npy(Matcha 依「音檔路徑往上兩層 / durations」讀時長),
這裡用 symlink 建一份獨立的音檔樹,X4 的清單指向這份音檔,時長也寫在這份樹底下:

    <out>/wav22k/<語者>/<檔名>.wav      → 主實驗 22 kHz 音檔的 symlink
    <out>/wav22k/durations/<檔名>.npy    (之後由 c2_mfa2durations.py --out-dir 寫入)
    <out>/lists/train.txt、val.txt       (主實驗 fair1h 清單,只把音檔路徑換成上面的 symlink)
    <out>/x4train_<腔>/、x4val_<腔>/      (MFA 用的 corpus:原始音檔 + .lab 的 symlink)

    python x4_prepare.py --dialect sixian --out ~/hakka_tts/exp_c/x4/sixian
"""
import argparse
import os
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from hakka_c import config as C  # noqa: E402


def link(src, dst):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if not dst.exists():
        os.symlink(Path(src).resolve(), dst)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dialect", required=True, choices=C.DIALECTS)
    ap.add_argument("--out", required=True)
    ap.add_argument("--data-yaml", help="預設 Matcha 的 configs/data/fair1h_joint_<腔>.yaml(主實驗 1 h 的訓練 / 驗證清單)")
    args = ap.parse_args()

    out = Path(args.out).expanduser()
    dy = Path(args.data_yaml).expanduser() if args.data_yaml else C.MATCHA / "configs" / "data" / f"fair1h_joint_{args.dialect}.yaml"
    cfg = yaml.safe_load(open(dy, encoding="utf-8"))
    corpus = {p.stem: p for p in (C.HAKKA / "data" / args.dialect / "corpus").rglob("*.wav")}
    if not corpus:
        sys.exit(f"★ {C.HAKKA}/data/{args.dialect}/corpus 底下沒有音檔")
    for split, key in (("train", "train_filelist_path"), ("val", "valid_filelist_path")):
        src = Path(cfg[key])
        src = src if src.is_absolute() else C.MATCHA / src
        lines, n_mfa = [], 0
        for line in open(src, encoding="utf-8"):
            line = line.rstrip("\n")
            if not line:
                continue
            wav, rest = line.split("|", 1)
            wav = Path(wav) if Path(wav).is_absolute() else C.MATCHA / wav
            stem, spk = wav.stem, wav.parent.name
            new = out / "wav22k" / spk / wav.name
            link(wav, new)
            lines.append(f"{new}|{rest}")
            if stem in corpus:                          # MFA corpus:原始音檔與 .lab
                w = corpus[stem]
                d = out / f"x4{split}_{args.dialect}" / w.parent.name
                link(w, d / w.name)
                if w.with_suffix(".lab").exists():
                    link(w.with_suffix(".lab"), d / f"{stem}.lab")
                n_mfa += 1
        (out / "lists").mkdir(parents=True, exist_ok=True)
        (out / "lists" / f"{split}.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"  {split}:{len(lines)} 句(來自 {src.name});MFA corpus {n_mfa} 句")


if __name__ == "__main__":
    main()
