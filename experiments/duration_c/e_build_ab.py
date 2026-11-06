"""實驗 E2:兩個系統的成對比較聽測(CMOS / AB 偏好),產生可以直接寄給聽者的資料夾。

    python e_build_ab.py --a ~/hakka_tts/exp_c/eval/base_sixian_step40000 \
                         --b ~/hakka_tts/exp_c/eval/c2_sixian_step40000 \
                         --n 30 --out ~/hakka_tts/exp_e/sixian_base_vs_c2 \
                         --submit-url https://script.google.com/macros/s/……/exec

輸出:
  <out>/listening_test/       ← 整個資料夾放到網頁空間(例如 Netlify Drop),把網址用 LINE 傳給聽者
      index.html              ← 手機點連結即可作答;給了 --submit-url 時每答一題就自動送到 Google 試算表(e_collect.gs)
      stimuli/tXX_1.wav, tXX_2.wav
  <out>/key.csv               ← 每題的「1/2 分別是哪個系統」,不要給聽者
  <out>/trials.json

設計(事先登記):
  * 句子:兩系統都有合成的測試句中,優先挑入聲 ≥ 2 個的句子,種子 20260716 抽 n 句;生成種子固定 1234。
  * 每題左右順序隨機;另外加入 n_identical 題「兩邊完全相同」的檢查題(期望回答 0),用來檢查聽者是否專心。
  * 響度:兩邊都正規化到相同 RMS(−23 dBFS),並限制峰值,避免「比較大聲就比較好」。
  * 聽者資料:會說的腔調可複選(四縣 / 海陸 / 其他),CSV 的 dialect 欄記成「四縣+海陸」這種形式。
  * 作答:7 點 CMOS,−3(1 好很多)… 0(一樣)… +3(2 好很多),外加「入聲字聽起來哪邊比較自然」的同一量表。
    畫面為長輩設計:大字、大按鈕、選項用文字(「第 1 段 比較好」);兩段都完整聽過才能作答;中途離開會自動接續。
"""
import argparse
import csv
import json
import random
import shutil
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from hakka_c import config as C  # noqa: E402


def rms_normalize(x, target_dbfs=-23.0, peak=0.95):
    x = np.asarray(x, np.float64)
    rms = np.sqrt(np.mean(x ** 2)) + 1e-12
    y = x * (10 ** (target_dbfs / 20) / rms)
    m = np.abs(y).max()
    return (y * (peak / m) if m > peak else y).astype(np.float32)


TEMPLATE = Path(__file__).resolve().parent / "e_ab_template.html"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True, help="系統 A 的 c_eval 輸出資料夾(含 wav/、bounds/)")
    ap.add_argument("--b", required=True, help="系統 B 的 c_eval 輸出資料夾")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--n-identical", type=int, default=2)
    ap.add_argument("--gen-seed", type=int, default=1234)
    ap.add_argument("--min-entering", type=int, default=2)
    ap.add_argument("--submit-url", default="", help="Google Apps Script 網頁應用程式網址(e_collect.gs);給了就自動回收作答")
    ap.add_argument("--label", help="顯示在標題的腔調名稱(預設依 --out 名稱判斷:sixian→四縣、hailu→海陸)")
    args = ap.parse_args()
    if args.submit_url and not args.submit_url.startswith("https://"):
        sys.exit("★ --submit-url 應該是 https://script.google.com/macros/s/…/exec")

    import soundfile as sf
    A, B = Path(args.a).expanduser(), Path(args.b).expanduser()
    out = Path(args.out).expanduser()
    stems = sorted({p.name.rsplit("_g", 1)[0] for p in (A / "wav").glob(f"*_g{args.gen_seed}.wav")}
                   & {p.name.rsplit("_g", 1)[0] for p in (B / "wav").glob(f"*_g{args.gen_seed}.wav")})
    if not stems:
        sys.exit("★ 兩個資料夾沒有共同的合成音檔:先對兩個系統都跑 c_eval.py --synth")

    def n_et(stem):
        b = json.load(open(A / "bounds" / f"{stem}.json", encoding="utf-8"))
        return sum(1 for s in b["syllables"] if s["entering"])

    rng = random.Random(C.SPLIT_SEED)
    rich = [s for s in stems if n_et(s) >= args.min_entering]
    pool = rich if len(rich) >= args.n else stems
    chosen = sorted(rng.sample(pool, min(args.n, len(pool))))
    ident = rng.sample(chosen, min(args.n_identical, len(chosen)))
    print(f"共同句子 {len(stems)} 句,入聲 ≥ {args.min_entering} 的 {len(rich)} 句;抽 {len(chosen)} 句 + {len(ident)} 題相同檢查題")

    if out.exists():
        shutil.rmtree(out)
    stim = out / "listening_test" / "stimuli"
    stim.mkdir(parents=True)
    items = [(s, "AB") for s in chosen] + [(s, "same") for s in ident]
    rng.shuffle(items)
    key, trials = [], []
    for k, (stem, kind) in enumerate(items, 1):
        tid = f"t{k:02d}"
        a, sr = sf.read(A / "wav" / f"{stem}_g{args.gen_seed}.wav")
        b, _ = sf.read(B / "wav" / f"{stem}_g{args.gen_seed}.wav")
        if kind == "same":
            pair, sys1, sys2 = (a, a), "A", "A"
        elif rng.random() < 0.5:
            pair, sys1, sys2 = (a, b), "A", "B"
        else:
            pair, sys1, sys2 = (b, a), "B", "A"
        for j, x in enumerate(pair, 1):
            sf.write(stim / f"{tid}_{j}.wav", rms_normalize(x), sr)
        key.append({"trial": tid, "stem": stem, "kind": kind, "pos1": sys1, "pos2": sys2,
                    "n_entering": n_et(stem)})
        trials.append({"id": tid})
    label = args.label if args.label is not None else ("四縣" if "sixian" in out.name else "海陸" if "hailu" in out.name else "")
    html = (TEMPLATE.read_text(encoding="utf-8").replace("__TRIALS__", json.dumps(trials)).replace("__TESTID__", out.name)
            .replace("__SUBMIT__", args.submit_url.replace("'", "")).replace("__LABEL__", label))
    (out / "listening_test" / "index.html").write_text(html, encoding="utf-8")
    with open(out / "key.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(key[0]))
        w.writeheader()
        w.writerows(key)
    json.dump({"A": str(A), "B": str(B), "gen_seed": args.gen_seed, "n": len(chosen), "n_identical": len(ident)},
              open(out / "trials.json", "w"), indent=2, ensure_ascii=False)
    print(f"→ 給聽者:{out / 'listening_test'}(把整個資料夾放到網頁空間,把網址傳給聽者)")
    print("   作答" + ("會自動送到 Google 試算表" if args.submit_url else "沒有自動回收:聽者要自己下載 / 分享結果(加 --submit-url 可自動回收)"))
    print(f"→ 自己保留:{out / 'key.csv'}(答案對照,不要給聽者)")
    print(f"   收回 CSV 後:python e_analyze.py --key {out / 'key.csv'} --responses <放 CSV 的資料夾>")


if __name__ == "__main__":
    main()
