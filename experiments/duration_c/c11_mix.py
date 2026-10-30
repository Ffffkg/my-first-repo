"""C11 第 1 步:合併「目標腔 1h(重複 R 次)」與「來源腔 48h」的 MFA 時長清單,當作聯合訓練的訓練清單。

兩份清單都是 Matcha 的 filelist(wav|語者|文字),wav 路徑是絕對路徑,durations/<檔名>.npy 在各自音檔旁邊,
所以直接串接就能用;語者編號是跨腔全域編號(四縣 0、1;海陸 2、3),不會互相覆蓋。
檢查:語者不重疊、抽樣檢查音檔與 durations 檔都在。

    python c11_mix.py --dialect sixian --source hailu --repeat 20
"""
import argparse
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from hakka_c import config as C  # noqa: E402


def read(path):
    return [l.rstrip("\n") for l in open(path, encoding="utf-8") if l.strip()]


def spk_of(lines):
    return {int(l.split("|")[1]) for l in lines}


def files_ok(lines, n=300, seed=0):
    pick = lines if len(lines) <= n else random.Random(seed).sample(lines, n)
    ok = 0
    for l in pick:
        w = Path(l.split("|")[0])
        w = w if w.is_absolute() else C.MATCHA / w
        ok += w.exists() and (w.parent.parent / "durations" / f"{w.stem}.npy").exists()
    return ok / len(pick)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dialect", required=True, choices=C.DIALECTS)
    ap.add_argument("--source", required=True, choices=C.DIALECTS)
    ap.add_argument("--repeat", type=int, default=20, help="目標腔清單重複幾次(決定目標腔在訓練樣本中的比例)")
    ap.add_argument("--target-list", default="ladder_1h_mfadur")
    ap.add_argument("--source-list", default="ladder_48h_mfadur")
    args = ap.parse_args()

    tdir = C.MATCHA / "data" / "hakka" / "joint" / args.dialect
    sdir = C.MATCHA / "data" / "hakka" / "joint" / args.source
    for f in (tdir / f"{args.target_list}.txt", sdir / f"{args.source_list}.txt"):
        if not f.exists():
            sys.exit(f"★ 找不到 {f}")
    t, s = read(tdir / f"{args.target_list}.txt"), read(sdir / f"{args.source_list}.txt")
    st, ss = spk_of(t), spk_of(s)
    print(f"目標腔 {args.dialect}:{len(t)} 句,語者 {sorted(st)};來源腔 {args.source}:{len(s)} 句,語者 {sorted(ss)}")
    if st & ss:
        sys.exit(f"★ 兩邊的語者編號重疊 {sorted(st & ss)}:直接串接會讓不同腔的語者共用同一個嵌入,請先重新編號")
    ft, fs = files_ok(t), files_ok(s)
    print(f"抽樣檢查音檔與 durations 都在的比例:目標 {ft:.1%}、來源 {fs:.1%}")
    if ft < 0.99 or fs < 0.99:
        sys.exit("★ 有音檔或 durations 檔不存在(來源腔 48h 的 MFA 時長要先產生:c2_mfa2durations.py --lists ladder_48h)")
    lines = t * args.repeat + s
    out = tdir / f"mix_1hx{args.repeat}_{args.source}48h_mfadur.txt"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    frac = len(t) * args.repeat / len(lines)
    print(f"已寫入 {out}\n  共 {len(lines)} 行(目標腔 {len(t)}×{args.repeat} = {len(t) * args.repeat},來源 {len(s)});目標腔佔 {frac:.0%}")
    print(f"  第一行(目標):{t[0]}\n  第一行(來源):{s[0]}")
    print(f"NAME={out.stem}")


if __name__ == "__main__":
    main()
