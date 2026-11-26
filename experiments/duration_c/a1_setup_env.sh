#!/usr/bin/env bash
# A1 前置:建立語音辨識(Omnilingual ASR)用的獨立 conda 環境 asr。不碰 tts 環境;可重複執行(裝好的會跳過),約 10 分鐘(不含模型下載)。
#   事先(要 sudo,只做一次;後面幾個是編譯 kenlm 用的):
#     sudo apt install -y build-essential cmake zlib1g-dev libbz2-dev liblzma-dev
#   bash a1_setup_env.sh
# 版本為什麼這樣釘:
#   * torch 2.8.0 + fairseq2 0.6.0:fairseq2 的原生函式庫必須和 PyTorch 版本完全對上(Python ≤ 3.12);omnilingual-asr 要求 fairseq2 在 0.5.2–0.6.0 之間。
#   * libsndfile 1.0.31(conda 版):fairseq2n 在 conda 環境裡不用系統的 libsndfile,否則 import 時報 "fairseq2 requires libsndfile"。
#   * pyarrow < 26:fairseq2 0.6 釘 numpy 1.x,而 pyarrow 26.0.0 起要求 NumPy 2("pyarrow requires NumPy 2.0 or newer")。
#   * transformers < 5:5.x 要求 huggingface_hub>=1.31,而 fairseq2 0.6 要求 huggingface_hub~=0.32。
#   * omnilingual-asr 從 GitHub 釘 commit 安裝:PyPI 上只有 0.1.0(只有 v1 模型卡 omniASR_LLM_1B 等),v2 模型卡(omniASR_LLM_1B_v2 等)只在 GitHub 版。
#   * kenlm(語言模型解碼用,我們用不到)在新版編譯器上可能編譯失敗;失敗時會自動改成不裝 kenlm 再試。
set -uo pipefail
source ~/miniconda3/etc/profile.d/conda.sh
act() { set +u; conda activate "$1"; set -u; }
must() { "$@" || { echo "★ 失敗:$*"; exit 1; }; }
step() { echo; echo "=== $* ==="; }
PYARROW='pyarrow>=20.0.0,<26'
FAIRSEQ2_INDEX=https://fair.pkg.atmeta.com/fairseq2/whl/pt2.8.0/cu128
OMNI_SRC="git+https://github.com/facebookresearch/omnilingual-asr.git@81f51e224ce9e74b02cc2a3eaf21b2d91d743455"   # 0.2.0(2025-12-30)

free_gb=$(df -BG --output=avail "$HOME" | tail -1 | tr -dc 0-9)
[ "$free_gb" -lt 30 ] && echo "★ 注意:家目錄只剩 ${free_gb} GB。環境約 8 GB、模型 8.5 GB(1B)到 17 GB(3B),建議至少留 30 GB。"

step "1/5 conda 環境 asr(Python 3.11、libsndfile 1.0.31)"
if conda env list | awk '{print $1}' | grep -x asr > /dev/null; then
  echo "環境 asr 已存在,沿用"
else
  must conda create -y -q -n asr --override-channels -c conda-forge python=3.11      # 只用 conda-forge:不會碰到 Anaconda 預設頻道的條款確認
fi
must conda install -y -q -n asr --override-channels -c conda-forge "libsndfile==1.0.31"
act asr

step "2/5 torch 2.8.0(cu128)"
must python -m pip install -q --upgrade pip
must pip install -q torch==2.8.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu128

step "3/5 fairseq2 0.6.0"
must pip install -q fairseq2==0.6.0 --extra-index-url "$FAIRSEQ2_INDEX"

step "4/5 omnilingual-asr"
install_omni() {   # $1 = pip 的來源。先完整安裝;失敗(多半是 kenlm 編譯失敗)就不裝相依套件,手動補上除了 kenlm 以外的
  pip install -q "$PYARROW" "$1" && return 0
  echo "★ 完整安裝失敗,改成不裝 kenlm 再試一次(只做辨識,用不到語言模型解碼)"
  pip install -q --no-deps "$1" && pip install -q "$PYARROW" numba pandas numpy "polars>=1.29.0" \
    "fairseq2[arrow]==0.6.0" --extra-index-url "$FAIRSEQ2_INDEX"
}
if ! install_omni "$OMNI_SRC"; then
  echo "★ 從 GitHub 安裝失敗,改用 PyPI 版(只有 v1 模型:跑的時候加  MODEL=omniASR_LLM_1B)"
  must install_omni omnilingual-asr
fi

step "5/5 其他套件(合成音檔轉 16k、報告、Whisper 後端)"
must pip install -q soundfile scipy numpy "transformers>=4.51,<5" "huggingface_hub>=0.34,<1.0" accelerate

step "檢查"
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
print("fairseq2", md.version("fairseq2"), "| omnilingual-asr", md.version("omnilingual-asr"),
      "| numpy", md.version("numpy"), "| pyarrow", md.version("pyarrow"), "| transformers", md.version("transformers"))
cards = pathlib.Path(omnilingual_asr.__file__).parent / "cards" / "models"
text = "".join(p.read_text(encoding="utf-8") for p in cards.glob("*.yaml"))
for name in ("omniASR_LLM_1B_v2", "omniASR_LLM_3B_v2", "omniASR_CTC_1B_v2", "omniASR_LLM_1B"):
    print(f"模型卡 {name}:", "有" if re.search(rf"^name:\s*{name}\s*$", text, re.M) else "沒有")
PY
echo "--- pip check(沒有列出東西就是沒有衝突)"
pip check
echo
echo "完成。接著:  DIALECTS=sixian NSENT=10 bash a1_run.sh   (先小試)"
