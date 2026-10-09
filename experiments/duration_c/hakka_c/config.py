"""路徑與常數。預設值取自教學手冊第 14.2 節的資料夾地圖。

路徑可以用環境變數覆寫,不必改程式:
    export HAKKA_ROOT=~/hakka_tts
    export MATCHA_ROOT=~/Matcha-TTS
"""
import os
from pathlib import Path

HAKKA = Path(os.environ.get("HAKKA_ROOT", "~/hakka_tts")).expanduser()
MATCHA = Path(os.environ.get("MATCHA_ROOT", "~/Matcha-TTS")).expanduser()
ANALYSIS = HAKKA / "analysis"          # 既有的自寫腳本(evaluate_v2.py 等)
BENCH = HAKKA / "benchmark"
VOCAB_JSON = BENCH / "hakka_matcha_vocab.json"
OUT_ROOT = HAKKA / "exp_c"             # 本實驗所有輸出

SR = 22050
HOP = 256
N_FFT = 1024
WIN = 1024
N_MELS = 80
F_MIN, F_MAX = 0, 8000
HOP_MS = HOP / SR * 1000.0             # ≈ 11.61 ms

DIALECTS = ("sixian", "hailu")
SPK_GENDER = {0: "F", 1: "M", 2: "F", 3: "M"}   # XF, XM, HF, HM
GEN_SEEDS = (1234, 2345, 3456, 4567, 5678)
N_TIMESTEPS = 10
TEMPERATURE = 0.667
SPLIT_SEED = 20260716

# 手冊主表(第 7.2 節)的最大資料量 baseline,用來做回歸核對
MAIN_TABLE_48H = {
    "sixian": {"et_vbias": 23.3, "nonet_vbias": 0.4},
    "hailu": {"et_vbias": 18.8, "nonet_vbias": -0.6},
}

VOWEL_CHARS = set("aeiou")
PAUSE_SYMBOLS = {"sp", "sil", "pau", "<sp>", "<sil>"}
MFA_SILENCE = {"", "sil", "sp", "<eps>", "silence"}
MFA_INVALID = {"spn", "<unk>"}          # 出現就整句跳過


def textgrid_dir(dialect):
    return HAKKA / "textgrids" / dialect


def filelist(mode, dialect, split):
    return MATCHA / "data" / "hakka" / mode / dialect / f"{split}.txt"


def exp_dir(exp):
    return MATCHA / "logs" / "train" / exp
