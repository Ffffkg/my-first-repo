#!/usr/bin/env bash
# A1 前置:建立語音辨識(Omnilingual ASR)用的獨立 conda 環境 asr。不碰 tts 環境;只要跑一次,約 10 分鐘(不含模型下載)。
#   事先(要 sudo,只做一次;libsndfile 是 fairseq2 需要的,後面幾個是編譯 kenlm 用的):
#     sudo apt install -y libsndfile1 build-essential cmake zlib1g-dev libbz2-dev liblzma-dev
#   bash a1_setup_env.sh
# 為什麼要獨立環境:fairseq2 的原生函式庫必須和 PyTorch 版本完全對上(fairseq2 0.6.0 ↔ torch 2.8.0,Python ≤ 3.12),
#   而 omnilingual-asr 要求 fairseq2 在 0.5.2–0.6.0 之間;裝進 tts 環境可能把 Matcha 的 torch 弄壞。
# 為什麼從 GitHub 裝:PyPI 上的 omnilingual-asr 只有 0.1.0(只有 v1 模型卡 omniASR_LLM_1B 等);
#   v2 模型卡(omniASR_LLM_1B_v2 等)只在 GitHub 的版本裡。
# kenlm(語言模型解碼用,我們用不到)在新版編譯器上可能編譯失敗;失敗時會自動改成不裝 kenlm 再試。
# libsndfile:fairseq2n 在 conda 環境裡不用系統的 libsndfile,一定要裝 conda 版的 1.0.31(否則 import 時報 "fairseq2 requires libsndfile")。
# transformers 釘在 4.x:5.x 要求 huggingface_hub>=1.31,而 fairseq2 0.6 要求 huggingface_hub~=0.32,裝了會衝突。
set -uo pipefail
source ~/miniconda3/etc/profile.d/conda.sh
act() { set +u; conda activate "$1"; set -u; }
must() { "$@" || { echo "★ 失敗:$*"; exit 1; }; }

free_gb=$(df -BG --output=avail "$HOME" | tail -1 | tr -dc 0-9)
[ "$free_gb" -lt 30 ] && echo "★ 注意:家目錄只剩 ${free_gb} GB。環境約 8 GB、模型 8.5 GB(1B)到 17 GB(3B),建議至少留 30 GB。"
ldconfig -p | grep -q libsndfile || echo "★ 找不到 libsndfile:先跑  sudo apt install -y libsndfile1"

if conda env list | awk '{print $1}' | grep -qx asr; then
  echo "環境 asr 已存在,沿用"
else
  must conda create -y -n asr --override-channels -c conda-forge python=3.11      # 只用 conda-forge:不會碰到 Anaconda 預設頻道的條款確認
fi
must conda install -y -n asr --override-channels -c conda-forge "libsndfile==1.0.31"
act asr
must python -m pip install --upgrade pip
must pip install torch==2.8.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu128
must pip install fairseq2==0.6.0 --extra-index-url https://fair.pkg.atmeta.com/fairseq2/whl/pt2.8.0/cu128

install_omni() {   # $1 = pip 的來源。先完整安裝;失敗(多半是 kenlm 編譯失敗)就不裝相依套件,手動補上除了 kenlm 以外的
  pip install "$1" && return 0
  echo "★ 完整安裝失敗,改成不裝 kenlm 再試一次(只做辨識,用不到語言模型解碼)"
  pip install --no-deps "$1" && pip install "pyarrow>=20.0.0" numba pandas numpy "polars>=1.29.0" \
    "fairseq2[arrow]==0.6.0" --extra-index-url https://fair.pkg.atmeta.com/fairseq2/whl/pt2.8.0/cu128
}
OMNI_SRC="git+https://github.com/facebookresearch/omnilingual-asr.git@81f51e224ce9e74b02cc2a3eaf21b2d91d743455"   # 0.2.0(2025-12-30),有 v2 模型卡
if ! install_omni "$OMNI_SRC"; then
  echo "★ 從 GitHub 安裝失敗,改用 PyPI 版(只有 v1 模型:跑的時候加  MODEL=omniASR_LLM_1B)"
  must install_omni omnilingual-asr
fi
must pip install soundfile scipy numpy "transformers>=4.51,<5" "huggingface_hub>=0.34,<1.0" accelerate    # 前三個:合成音檔轉 16k、報告;後面:Whisper 後端

must python - <<'PY'
import ctypes.util
import importlib.metadata as md
import pathlib
import re

import omnilingual_asr
import torch
from omnilingual_asr.models.inference.pipeline import ASRInferencePipeline  # noqa: F401

print("torch", torch.__version__, "| cuda", torch.cuda.is_available(),
      torch.cuda.get_device_name(0) if torch.cuda.is_available() else "")
if torch.cuda.is_available():
    x = torch.randn(64, 64, device="cuda")
    print("GPU 運算測試:", float((x @ x).sum().isfinite()))        # RTX 50 系(sm_120)要 cu128 的 torch 才跑得動
print("fairseq2", md.version("fairseq2"), "| omnilingual-asr", md.version("omnilingual-asr"))
print("libsndfile:", ctypes.util.find_library("sndfile") or "★ 找不到(sudo apt install -y libsndfile1)")
cards = pathlib.Path(omnilingual_asr.__file__).parent / "cards" / "models"
text = "".join(p.read_text(encoding="utf-8") for p in cards.glob("*.yaml"))
for name in ("omniASR_LLM_1B_v2", "omniASR_LLM_3B_v2", "omniASR_CTC_1B_v2", "omniASR_LLM_1B"):
    print(f"模型卡 {name}:", "有" if re.search(rf"^name:\s*{name}\s*$", text, re.M) else "沒有")
PY
pip check || echo "(上面若有套件相依衝突,貼給我看;只有 kenlm 缺的話可以忽略)"
echo
echo "完成。接著:  DIALECTS=sixian NSENT=10 bash a1_run.sh   (先小試)"
