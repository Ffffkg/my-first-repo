#!/usr/bin/env bash
# R0:安裝 Raon-OpenTTS(第三種架構)。只需要跑一次,約 10–20 分鐘(看網路)。
#   bash r_setup.sh
# 做的事:
#   1. 下載官方程式碼到 ~/Raon-OpenTTS(固定在 2026-05-22 的版本)
#   2. 建立 conda 環境 raon(複製 tts 環境,不動到 tts;torch 版本鎖定不變)
#   3. 下載官方 0.3B 權重(KRAFTON/Raon-OpenTTS-0.3B)與 16 kHz HiFi-GAN 聲碼器到 ~/hakka_tts/raon/pretrained/
#   4. 在 CPU 上建模型、載入權重,確認詞表大小與權重相符
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
H=${HAKKA_ROOT:-$HOME/hakka_tts}; RAON=${RAON_ROOT:-$H/raon}; REPO=${RAON_REPO:-$HOME/Raon-OpenTTS}
COMMIT=0dd28405c9348e9505c2a1c92a250cb3beffd950
source ~/miniconda3/etc/profile.d/conda.sh
act() { set +u; conda activate "$1"; set -u; }

free_gb=$(df -BG --output=avail "$HOME" | tail -1 | tr -dc 0-9)
echo "家目錄剩餘空間 ${free_gb} GB(環境約 10 GB、權重與聲碼器約 2–5 GB、每次訓練的存檔約 4 GB)"
[ "$free_gb" -ge 30 ] || { echo "★ 剩餘空間不到 30 GB,先清出空間"; exit 1; }

echo "===== 1. 官方程式碼"
[ -d "$REPO/.git" ] || git clone -q https://github.com/krafton-ai/RAON-OpenTTS.git "$REPO"
git -C "$REPO" fetch -q origin "$COMMIT" 2>/dev/null || true
git -C "$REPO" checkout -q "$COMMIT"
echo "  $REPO @ $(git -C "$REPO" log -1 --format='%h %ci')"

echo "===== 2. conda 環境 raon"
if ! conda env list | awk '{print $1}' | grep -qx raon; then
  conda create -y -q -n raon --clone tts > /dev/null
fi
act raon
python - > /tmp/raon_constraints.txt <<'PY'
import importlib.metadata as md
for p in ("torch", "torchaudio", "numpy"):
    try:
        print(f"{p}=={md.version(p)}")
    except md.PackageNotFoundError:
        pass
PY
echo "  鎖定版本:$(tr '\n' ' ' < /tmp/raon_constraints.txt)"
pip install -q -c /tmp/raon_constraints.txt "x-transformers>=2.0.0" torchdiffeq "ema-pytorch>=0.7.0" "einops>=0.8.0" \
    safetensors accelerate wandb hydra-core omegaconf pydub jieba pypinyin "huggingface_hub>=0.23" librosa pyarrow
# 不用 pip install -e(官方 pyproject 的建置設定較少見),直接把 src 加進這個環境的搜尋路徑
echo "$REPO/src" > "$(python -c 'import site; print(site.getsitepackages()[0])')/raon_opentts.pth"
python -c "import torch, f5_tts.model.cfm; print('  torch', torch.__version__, 'CUDA', torch.cuda.is_available())"

echo "===== 3. 官方權重與聲碼器"
mkdir -p "$RAON/pretrained"
python - "$RAON/pretrained" <<'PY'
import sys
from pathlib import Path
from huggingface_hub import hf_hub_download, snapshot_download
root = Path(sys.argv[1])
snapshot_download("KRAFTON/Raon-OpenTTS-0.3B", local_dir=root / "Raon-OpenTTS-0.3B")
hf_hub_download("speechbrain/tts-hifigan-libritts-16kHz", "generator.ckpt", local_dir=root / "hifigan16k")
for p in sorted(root.rglob("*")):
    if p.is_file() and ".cache" not in p.parts:
        print(f"  {p.relative_to(root)}  {p.stat().st_size / 2**20:.0f} MB")
PY

echo "===== 4. 檢查:建模型、載入權重(CPU)"
cd "$HERE"
python - <<'PY'
from hakka_c import raon_io as R
ck, vocab = R.find_pretrained()
m = R.read_vocab(vocab)
model = R.build_model(m, "cpu")
import string
R.load_pretrained(model, ck, len(m), used_chars=string.ascii_lowercase + string.digits + " ")
print(f"  checkpoint {ck.name}、詞表 {len(m)} 字元、參數 {sum(p.numel() for p in model.parameters()) / 1e6:.0f}M ✓")
voc = R.load_vocoder("cpu")
print("  聲碼器 ✓")
PY
echo "完成。下一步:nohup bash r_run.sh > ~/hakka_tts/runs/r.log 2>&1 &"
