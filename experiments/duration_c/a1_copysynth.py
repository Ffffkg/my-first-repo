"""A1 天花板:測試集的真人錄音,用它自己的真實 mel 直接過同一個 HiFi-GAN 聲碼器重建(分析再合成,copy-synthesis)。

這是「聲學模型與時長都完美」時、經過同一個聲碼器的語音。拿去給 ASR 辨識,放進 A1 報告當 copy 那一列,
就知道各項指標(入聲保留率、聲調記號一致率…)最好能到多少——合成系統的數字要跟它比,才知道還有多少進步空間。

    python a1_copysynth.py --dialect sixian --like $E/base_sixian_step40000 --out ~/hakka_tts/exp_c/a1/copysynth_sixian
輸出 <out>/wav/<stem>_g1234.wav(檔名格式同 D1 評估資料夾,可直接給 a1_asr_check.py run --sys copy=<out>)。
需要 tts 環境(torch + Matcha),用 GPU 約幾十秒。
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from hakka_c import config as C  # noqa: E402
from hakka_c.matcha_io import mel_spectrogram, read_filelist  # noqa: E402  (匯入時會把 Matcha 加進 sys.path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dialect", required=True, choices=C.DIALECTS)
    ap.add_argument("--like", required=True, help="D1 評估資料夾:取它 wav/ 底下的句子清單,讓天花板和各系統用同一批句子")
    ap.add_argument("--out", required=True)
    ap.add_argument("--gen-seed", type=int, default=C.GEN_SEEDS[0], help="只影響輸出檔名(重建沒有隨機性)")
    ap.add_argument("--vocoder", help="HiFi-GAN universal v1 checkpoint(預設 Matcha 的下載位置)")
    args = ap.parse_args()

    import soundfile as sf
    import torch
    from matcha.cli import load_hifigan
    from matcha.utils.utils import get_user_data_dir

    suffix = f"_g{args.gen_seed}.wav"
    like = Path(args.like).expanduser()
    stems = sorted(p.name[:-len(suffix)] for p in (like / "wav").glob(f"*{suffix}"))
    real = {Path(r["wav"]).stem: r["wav"] for r in read_filelist(C.filelist("joint", args.dialect, "test"))}
    stems = [s for s in stems if s in real]
    if not stems:
        sys.exit(f"★ {like}/wav 底下沒有 *{suffix},或這些句子不在 {args.dialect} 的測試清單裡")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vocoder = load_hifigan(str(Path(args.vocoder) if args.vocoder else get_user_data_dir() / "hifigan_univ_v1"), device)
    out = Path(args.out).expanduser() / "wav"
    out.mkdir(parents=True, exist_ok=True)
    with torch.inference_mode():
        for i, st in enumerate(stems, 1):
            a, sr = sf.read(real[st], dtype="float32", always_2d=True)
            assert sr == C.SR, f"{real[st]} 取樣率 {sr} != {C.SR}"
            x = torch.from_numpy(a.mean(axis=1))[None].to(device)
            mel = mel_spectrogram(x, C.N_FFT, C.N_MELS, C.SR, C.HOP, C.WIN, C.F_MIN, C.F_MAX, center=False)    # 與 Engine.mel() 相同,不做正規化
            wav = vocoder(mel).clamp(-1, 1).squeeze().float().cpu().numpy()                                   # 與 Engine.to_wav() 相同
            sf.write(out / f"{st}{suffix}", wav, C.SR)
            if i % 25 == 0 or i == len(stems):
                print(f"  已重建 {i}/{len(stems)}")
    print(f"→ {out}({len(stems)} 句)")


if __name__ == "__main__":
    main()
