"""R0b:確認字元編號對得上官方權重——用「沒有微調」的官方模型唸英文,人耳聽一下。需要 raon 環境與 GPU,約 1 分鐘。

官方 vocab.txt 比權重多 4 個字元(r_train.py 會印出推論:多的字元不在 ASCII 區,客語拼音用到的編號不受影響)。
這裡做直接驗證,輸出兩個檔:
  sanity_en_correct.wav   用目前的字元編號 → 應該是清楚的英文
  sanity_en_shift1.wav    故意把 ASCII 編號錯一格 → 應該是聽不懂的聲音
如果兩個聽起來一樣糊,或 shift1 反而比較清楚,代表編號有錯,先不要訓練。
聲音提示用四縣語者 0 的提示句(客語),所以口音會怪,只要聽得出英文單字就對了。

    python r_sanity_en.py
"""
import json
import string
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from hakka_c import config as C  # noqa: E402
from hakka_c import raon_io as R  # noqa: E402

TEXT = "the quick brown fox jumps over the lazy dog. please call stella and ask her to bring these things."


def main():
    import soundfile as sf
    ck, vocab_path = R.find_pretrained()
    m = R.read_vocab(vocab_path)
    model = R.build_model(m, "cuda")
    R.load_pretrained(model, ck, len(m), used_chars=string.ascii_lowercase + string.digits + " .")
    model.eval()
    voc = R.load_vocoder("cuda")
    p = json.load(open(R.data_dir("sixian") / "prompts.json", encoding="utf-8"))["0"]
    audio = sf.read(p["wav16"], dtype="float32")[0]
    gen = round(len(TEXT) / 14.0 * R.FPS16)                  # 英文約每秒 14 個字元
    out = C.OUT_ROOT / "r"
    out.mkdir(parents=True, exist_ok=True)
    ascii_n = sum(1 for t in m if len(t) == 1 and ord(t) < 128)
    shifted = {t: (i + 1 if i < ascii_n - 1 else i) for t, i in m.items()}
    for lab, vm in (("correct", m), ("shift1", shifted)):
        model.vocab_char_map = vm
        w = R.synthesize(model, voc, audio, p["text"], TEXT, gen, 1234)
        sf.write(out / f"sanity_en_{lab}.wav", w, R.SR16)
        print(f"→ {out / f'sanity_en_{lab}.wav'}")


if __name__ == "__main__":
    main()
