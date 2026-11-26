"""R4:在相同頻寬下重算 MCD,讓 16 kHz 的 Raon-OpenTTS 能和 22.05 kHz 的系統公平比較。不需要 GPU。

原本的 MCD(mcd.csv)是在 22.05 kHz 算的:真人錄音與 Matcha-TTS / VITS 有 8–11 kHz 的頻帶,Raon-OpenTTS(16 kHz)
沒有,所以 Raon 的 MCD 會多出約 4 dB,反映的是頻寬而不是音質。
這裡把「合成語音」與「真人錄音」都先降到 16 kHz 再升回 22.05 kHz(等於濾掉 8 kHz 以上),
再用與論文相同的 evaluate_v2.py(analyze_wav、mcd_from_mcep)重算。每個評估資料夾寫一個 mcd16.csv(欄位同 mcd.csv),
報告用 x1_overlap.py --mcd-csv mcd16.csv。
另外抽前 5 個音檔不濾波重算一次,和 mcd.csv 比對,確認計算流程與原本相同(差異只來自音檔存成 16 位元)。

    python r_mcd16.py --dialect sixian --dirs DIR1 DIR2 ... --jobs 8
"""
import argparse
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from hakka_c import config as C  # noqa: E402
from hakka_c.csvio import read_cols, write_rows  # noqa: E402

EV = None


def init():
    global EV
    sys.path.insert(0, str(C.ANALYSIS))
    import evaluate_v2
    EV = evaluate_v2


def band16(x):
    """22.05 kHz → 16 kHz → 22.05 kHz(濾掉 8 kHz 以上),長度不變。"""
    from scipy.signal import resample_poly
    y = resample_poly(resample_poly(x, 320, 441), 441, 320)
    return np.pad(y, (0, max(0, len(x) - len(y))))[: len(x)]


def read(path):
    import soundfile as sf
    a, sr = sf.read(str(path), dtype="float64", always_2d=True)
    assert sr == C.SR, (path, sr)
    return a.mean(1)


def mcep(job):
    path, filt = job
    x = read(path)
    return EV.analyze_wav(band16(x) if filt else x)[1]


def mcd_job(job):
    path, ref_mc, filt = job
    return float(EV.mcd_from_mcep(mcep((path, filt)), ref_mc))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dialect", required=True, choices=C.DIALECTS)
    ap.add_argument("--dirs", nargs="+", required=True)
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    real = {}
    for line in open(C.filelist("joint", args.dialect, "test"), encoding="utf-8"):
        if line.strip():
            w = Path(line.split("|")[0])
            real[w.stem] = w if w.is_absolute() else C.MATCHA / w
    dirs = [Path(d).expanduser() for d in args.dirs]
    todo = [d for d in dirs if args.force or not (d / "mcd16.csv").exists()]
    print(f"{len(dirs)} 個資料夾,{len(todo)} 個要算")
    if not todo:
        return
    rows = {d: read_cols(d / "mcd.csv") for d in todo}
    stems = sorted({s for c in rows.values() for s in np.asarray(c["stem"]).astype(str)})
    with Pool(args.jobs, initializer=init) as pool:
        ref = dict(zip(stems, pool.map(mcep, [(real[s], True) for s in stems])))
        ref_raw = None
        for d in todo:
            c = rows[d]
            st = np.asarray(c["stem"]).astype(str)
            gs = c["gen_seed"].astype(int)
            wavs = [d / "wav" / f"{s}_g{g}.wav" for s, g in zip(st, gs)]
            if ref_raw is None:                     # 檢查:不濾波時應重現 mcd.csv
                chk = list(range(min(5, len(wavs))))
                rr = dict(zip(st[chk], pool.map(mcep, [(real[s], False) for s in st[chk]])))
                got = pool.map(mcd_job, [(wavs[i], rr[st[i]], False) for i in chk])
                diff = np.abs(np.array(got) - c["mcd"][chk].astype(float))
                print(f"  檢查(不濾波重算 vs mcd.csv,{d.name}):最大差 {diff.max():.3f} dB")
                if diff.max() > 0.2:
                    sys.exit("★ 重算結果和 mcd.csv 差太多,計算流程可能不同,先停下來")
                ref_raw = rr
            m = pool.map(mcd_job, [(w, ref[s], True) for w, s in zip(wavs, st)], chunksize=8)
            out = [{"stem": s, "spk": int(sp), "gender": g, "gen_seed": int(gg), "mcd": v}
                   for s, sp, g, gg, v in zip(st, c["spk"], np.asarray(c["gender"]).astype(str), gs, m)]
            write_rows(d / "mcd16.csv", out)
            print(f"  {d.name}:8 kHz 以下 MCD 平均 {np.mean(m):.3f} dB(原本 {c['mcd'].astype(float).mean():.3f})", flush=True)


if __name__ == "__main__":
    main()
