"""實驗 C2 第 1 步:MFA TextGrid → Matcha 的 precomputed durations(手冊第 25.5 節)。

Matcha 0.0.7 的 TextMelDataset 在 load_durations: true 時,會讀
    <音檔路徑>.parent.parent / "durations" / <檔名>.npy
長度必須等於 intersperse 後的符號數 2n+1,總和必須等於梅爾長度(= 22 kHz 樣本數 // 256)。

★ 最大的坑:TextGrid 是在「原始 48 kHz、未裁切」的音檔上對齊的,22 kHz 訓練音檔已被
  resample_audio.py 裁掉句首靜音。母音「時長」不受影響,但這裡要的是「絕對位置」,
  所以必須知道每句的裁切起點(offset)。四種取得方式:
    --offset-mode zero    TextGrid 本來就是在 22 kHz 裁切後音檔上對齊的(例如你重跑過 MFA)
    --offset-mode csv     resample_audio.py 有記錄:--offset-csv 檔案每行 stem,offset_秒
    --offset-mode margin  依裁切規則推算:offset = 第一個音素起點 − margin(--margin-ms,看 resample_audio.py)
    --offset-mode xcorr   用原始音檔做互相關自動對位(--orig-root 原始 wav 根目錄;最穩,但較慢)
  不論哪種,程式都會檢查推算後的句首/句尾留白是否合理,不合理的句子排除並列出。

    python c2_mfa2durations.py --dialect sixian --offset-mode xcorr --orig-root ~/HAT/tts_sixian --dry-run --limit 50
    python c2_mfa2durations.py --dialect sixian --offset-mode xcorr --orig-root ~/HAT/tts_sixian
"""
import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from hakka_c import config as C  # noqa: E402
from hakka_c.durations import merge_blank, mfa_to_frames  # noqa: E402
from hakka_c.structure import StructError, build_utt, is_pause, load_pua_map  # noqa: E402
from hakka_c.textgrid import read_textgrid  # noqa: E402


def xcorr_offset(trim, orig, sr_orig, max_off_s=8.0, win_s=3.0):
    """回傳 (offset_秒, 正規化相關係數)。trim 為 22 kHz 裁切後音檔。"""
    from math import gcd

    from scipy.signal import correlate, resample_poly
    g = gcd(C.SR, sr_orig)
    o = resample_poly(orig, C.SR // g, sr_orig // g)
    w = trim[: int(win_s * C.SR)]
    seg = o[: len(w) + int(max_off_s * C.SR)]
    if len(seg) < len(w):
        return None, 0.0
    num = correlate(seg, w, mode="valid", method="fft")
    e = np.convolve(seg ** 2, np.ones(len(w)), mode="valid")
    den = np.sqrt(np.maximum(e, 1e-12) * np.sum(w ** 2))
    r = num / den
    k = int(np.argmax(r))
    return k / C.SR, float(r[k])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dialect", required=True, choices=C.DIALECTS)
    ap.add_argument("--mode", default="joint", choices=("joint", "separate"))
    ap.add_argument("--lists", nargs="*", help="filelist 名稱(不含 .txt);預設資料夾內全部")
    ap.add_argument("--offset-mode", required=True, choices=("zero", "csv", "margin", "xcorr"))
    ap.add_argument("--offset-csv")
    ap.add_argument("--margin-ms", type=float)
    ap.add_argument("--orig-root", help="原始(未裁切)wav 的根目錄")
    ap.add_argument("--max-lead-ms", type=float, default=600, help="推算後句首留白上限")
    ap.add_argument("--min-lead-ms", type=float, default=-15, help="容許第一個音素稍早於 0 的誤差")
    ap.add_argument("--out-dir", help="預設依 Matcha 慣例:<音檔>.parent.parent/durations")
    ap.add_argument("--cleaners", default="hakka_cleaners")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--dry-run", action="store_true", help="只檢查、不寫檔")
    args = ap.parse_args()

    import soundfile as sf
    sys.path.insert(0, str(C.MATCHA))
    from matcha.text import text_to_sequence

    pua = load_pua_map(C.VOCAB_JSON)
    d_dir = C.filelist(args.mode, args.dialect, "x").parent
    names = args.lists or sorted(p.stem for p in d_dir.glob("*.txt") if not p.stem.endswith("_mfadur"))
    tg_idx = {p.stem: p for p in C.textgrid_dir(args.dialect).rglob("*.TextGrid")}
    offsets = {}
    if args.offset_mode == "csv":
        for row in csv.reader(open(args.offset_csv, encoding="utf-8")):
            if row and row[0] != "stem":
                offsets[row[0]] = float(row[1])
    orig_idx = {}
    if args.offset_mode == "xcorr":
        orig_idx = {p.stem: p for p in Path(args.orig_root).expanduser().rglob("*.wav")}
        print(f"原始音檔 {len(orig_idx)} 個")
    if args.offset_mode == "margin" and args.margin_ms is None:
        sys.exit("--offset-mode margin 需要 --margin-ms(看 resample_audio.py 的 margin 設定)")

    done, report, leads, tails, why = {}, {}, [], [], Counter()
    for name in names:
        fl = d_dir / f"{name}.txt"
        lines = [l.rstrip("\n") for l in open(fl, encoding="utf-8") if l.strip()]
        if args.limit:
            lines = lines[: args.limit]
        kept = []
        for line in lines:
            wav, spk, text = line.split("|")[:3]
            stem = Path(wav).stem
            if stem in done:
                if done[stem]:
                    kept.append(line)
                continue
            try:
                _, clean = text_to_sequence(text, [args.cleaners])
                sym = [pua[c] for c in clean]
                if stem not in tg_idx:
                    raise StructError("找不到 TextGrid")
                utt = build_utt(stem, sym, read_textgrid(tg_idx[stem]))
                info = sf.info(wav)
                n_samp = info.frames
                T = n_samp // C.HOP
                if args.offset_mode == "zero":
                    off = 0.0
                elif args.offset_mode == "csv":
                    if stem not in offsets:
                        raise StructError("offset csv 沒有這句")
                    off = offsets[stem]
                elif args.offset_mode == "margin":
                    off = max(0.0, utt.phone_iv[0][0] - args.margin_ms / 1000)
                else:
                    if stem not in orig_idx:
                        raise StructError("找不到原始音檔")
                    a, _ = sf.read(wav, dtype="float32", always_2d=True)
                    o, sr_o = sf.read(str(orig_idx[stem]), dtype="float32", always_2d=True)
                    off, r = xcorr_offset(a.mean(1), o.mean(1), sr_o)
                    if off is None or r < 0.8:
                        raise StructError(f"互相關太低 ({r:.2f})")
                lead = (utt.phone_iv[0][0] - off) * 1000
                tail = (n_samp / C.SR - (utt.phone_iv[-1][1] - off)) * 1000
                if not (args.min_lead_ms <= lead <= args.max_lead_ms) or tail < args.min_lead_ms:
                    raise StructError("推算後的句首/句尾留白不合理")
                pause = [i for i, nm in enumerate(sym) if is_pause(nm)]
                fr = mfa_to_frames(len(sym), utt.phone_sym, utt.phone_iv, T, pause, off)
                # 自我檢查:左右 blank 都只有 1 frame 的內部音素,經 merge_blank(half) 應還原成 MFA frame 數(±1)
                mb = merge_blank(fr, "half")
                for k, i in enumerate(utt.phone_sym[1:-1], start=1):
                    if fr[2 * i] != 1 or fr[2 * i + 2] != 1:
                        continue
                    a_ = utt.phone_iv[k]
                    want = round((a_[1] - off) * C.SR / C.HOP) - round((a_[0] - off) * C.SR / C.HOP)
                    if abs(mb[i] - max(want, 1)) > 1.01:
                        raise StructError("merge 還原檢查失敗")
                leads.append(lead)
                tails.append(tail)
                if not args.dry_run:
                    out_dir = Path(args.out_dir).expanduser() if args.out_dir else Path(wav).parent.parent / "durations"
                    out_dir.mkdir(parents=True, exist_ok=True)
                    np.save(out_dir / f"{stem}.npy", fr.astype(np.int64))
                done[stem] = True
                kept.append(line)
            except (StructError, KeyError, ValueError) as e:
                done[stem] = False
                why[str(e)[:60]] += 1
        report[name] = {"total": len(lines), "kept": len(kept)}
        print(f"{name}: {len(kept)}/{len(lines)} 句可用")
        if not args.dry_run:
            (d_dir / f"{name}_mfadur.txt").write_text("\n".join(kept) + "\n", encoding="utf-8")

    if leads:
        q = lambda v: ", ".join(f"{x:.0f}" for x in np.percentile(v, [1, 5, 50, 95, 99]))  # noqa: E731
        print(f"\n推算後句首留白 (ms) 1/5/50/95/99 百分位:{q(leads)}")
        print(f"推算後句尾留白 (ms) 1/5/50/95/99 百分位:{q(tails)}")
        print("→ 兩者應集中在 resample_audio.py 保留的 margin 附近;若分布很散,offset 推算方式可能不對。")
    print("排除原因:", dict(why.most_common()) or "無")
    if not args.dry_run:
        json.dump({"args": vars(args), "lists": report, "excluded": dict(why)},
                  open(d_dir / "mfadur_report.json", "w"), indent=2, ensure_ascii=False)
        print(f"\n→ durations 已寫入;過濾後的清單 {d_dir}/*_mfadur.txt;報告 {d_dir}/mfadur_report.json")
        print("  下一步:python c2_setup.py --base-exp <你的 baseline experiment 名稱>")


if __name__ == "__main__":
    main()
