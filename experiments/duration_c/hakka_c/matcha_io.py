"""與 Matcha-TTS 互動的部分(需要 tts 環境:torch + Matcha)。

設計原則:不改 Matcha 原始碼。MAS 與合成都在這裡「照 Matcha 0.0.7 的寫法重做一遍」,
差別只在於可以換掉時長(例如換取整方式、乘縮放係數)。selfcheck() 會驗證在不換時長時,
這裡的合成結果與 model.synthesise() 逐點相同。
"""
import math
import re
import sys
from pathlib import Path

import numpy as np

from . import config as C
from .structure import load_pua_map


def add_paths():
    for p in (C.MATCHA, C.ANALYSIS):
        if p.exists() and str(p) not in sys.path:
            sys.path.insert(0, str(p))


add_paths()

import soundfile as sf  # noqa: E402
import torch  # noqa: E402
from matcha.models.matcha_tts import MatchaTTS  # noqa: E402
from matcha.text import text_to_sequence  # noqa: E402
from matcha.utils.audio import mel_spectrogram  # noqa: E402
from matcha.utils.model import fix_len_compatibility, generate_path, sequence_mask  # noqa: E402
from matcha.utils.utils import intersperse  # noqa: E402
import matcha.utils.monotonic_align as monotonic_align  # noqa: E402


# ---------------------------------------------------------------- 檔案
def read_filelist(path):
    rows = []
    for line in open(path, encoding="utf-8"):
        line = line.rstrip("\n")
        if not line:
            continue
        parts = line.split("|")
        rows.append({"wav": parts[0], "spk": int(parts[1]), "text": parts[2], "line": line})
    return rows


def find_ckpt(exp, which="best"):
    """與 pick_ckpt() 的修正版相同:只看最新一次 run;best = 檔名上驗證損失最低者。"""
    runs = sorted((C.exp_dir(exp) / "runs").glob("*/checkpoints"))
    if not runs:
        raise FileNotFoundError(f"找不到 {C.exp_dir(exp)}/runs/*/checkpoints")
    ck = runs[-1]
    if which == "best":
        bests = list(ck.glob("best*.ckpt"))
        if not bests:
            raise FileNotFoundError(f"{ck} 沒有 best*.ckpt")

        def loss(p):  # best_{epoch}_{step}_{val_loss}.ckpt
            m = re.search(r"_(\d+\.\d+)$", p.stem)
            return float(m.group(1)) if m else 1e9

        return min(bests, key=loss)
    if which == "last":
        return ck / "last.ckpt"
    m = re.fullmatch(r"step(\d+)", which)
    if m:
        return ck / f"step_{int(m.group(1)):07d}.ckpt"
    return Path(which)


def textgrid_index(dialect):
    idx = {}
    for p in C.textgrid_dir(dialect).rglob("*.TextGrid"):
        idx[p.stem] = p
    if not idx:
        raise FileNotFoundError(f"{C.textgrid_dir(dialect)} 底下沒有 TextGrid")
    return idx


def load_matcha(ckpt, device):
    """torch ≥ 2.6 的 torch.load 預設 weights_only=True,會拒絕含 OmegaConf 超參數的 Matcha checkpoint。
    依序嘗試:Lightning 的 weights_only=False → 手動 torch.load(weights_only=False) 後重建模型。
    (checkpoint 是你自己訓練的,可信任。)"""
    try:
        return MatchaTTS.load_from_checkpoint(str(ckpt), map_location=device, weights_only=False)
    except TypeError:                       # 舊版 Lightning 沒有 weights_only 參數
        pass
    except Exception as e:  # noqa: BLE001
        if "weights_only" not in str(e) and "WeightsUnpickler" not in str(e):
            raise
    try:
        return MatchaTTS.load_from_checkpoint(str(ckpt), map_location=device)
    except Exception as e:  # noqa: BLE001
        if "weights_only" not in str(e) and "WeightsUnpickler" not in str(e):
            raise
    state = torch.load(str(ckpt), map_location=device, weights_only=False)
    model = MatchaTTS(**state["hyper_parameters"])
    model.load_state_dict(state["state_dict"])
    return model


def load_mel_stats(data_yaml):
    import yaml
    cfg = yaml.safe_load(open(data_yaml, encoding="utf-8"))
    ds = cfg["data_statistics"]
    return float(ds["mel_mean"]), float(ds["mel_std"])


# ---------------------------------------------------------------- 模型
class Engine:
    def __init__(self, ckpt, device="cuda", cleaners=("hakka_cleaners",), mel_stats=None, vocab_json=None,
                 model=None):
        self.device = torch.device(device if torch.cuda.is_available() or device == "cpu" else "cpu")
        if model is None:
            model = load_matcha(ckpt, self.device)
        self.model = model.to(self.device).eval()
        self.cleaners = list(cleaners)
        if ckpt is None and vocab_json is None:
            self.pua = {}                     # 測試用:直接給 model、不處理文字
        else:
            self.pua = load_pua_map(vocab_json or C.VOCAB_JSON)
        if mel_stats is None:
            mel_stats = (float(self.model.mel_mean), float(self.model.mel_std))
            self.stats_src = "model buffer"
        else:
            self.stats_src = "data config"
        self.mel_mean, self.mel_std = mel_stats
        self.vocoder = None
        self.ckpt = ckpt

    # 文字 ------------------------------------------------------------
    def text(self, text):
        seq, clean = text_to_sequence(text, self.cleaners)
        names = []
        for ch in clean:
            if ch not in self.pua:
                raise KeyError(f"PUA 字元 U+{ord(ch):04X} 不在對照表中")
            names.append(self.pua[ch])
        x = intersperse(seq, 0)
        return torch.tensor(x, dtype=torch.long, device=self.device)[None], names

    def spk(self, spk_id):
        if self.model.n_spks > 1:
            return self.model.spk_emb(torch.tensor([spk_id], device=self.device).long())
        return None

    @torch.inference_mode()
    def encode(self, x, spk_id):
        x_len = torch.tensor([x.shape[1]], device=self.device)
        spks = self.spk(spk_id)
        mu_x, logw, x_mask = self.model.encoder(x, x_len, spks)
        w = (torch.exp(logw) * x_mask)[0, 0].float().cpu().numpy()
        return mu_x, x_mask, spks, w

    # 音訊 ------------------------------------------------------------
    def wav(self, path):
        a, sr = sf.read(str(path), dtype="float32", always_2d=True)
        assert sr == C.SR, f"{path} 取樣率 {sr} != {C.SR}"
        return a.mean(axis=1)

    @torch.inference_mode()
    def mel(self, path):
        a = torch.from_numpy(self.wav(path))[None].to(self.device)
        m = mel_spectrogram(a, C.N_FFT, C.N_MELS, C.SR, C.HOP, C.WIN, C.F_MIN, C.F_MAX, center=False)
        return (m - self.mel_mean) / self.mel_std           # (1, 80, T)

    @torch.inference_mode()
    def mas(self, mu_x, x_mask, y):
        """teacher-forced 單調對齊:與 MatchaTTS.forward() 裡 MAS 的程式相同。"""
        T = y.shape[-1]
        y_mask = sequence_mask(torch.tensor([T], device=self.device), T).unsqueeze(1).to(x_mask)
        attn_mask = x_mask.unsqueeze(-1) * y_mask.unsqueeze(2)
        n_feats = self.model.n_feats
        const = -0.5 * math.log(2 * math.pi) * n_feats
        factor = -0.5 * torch.ones(mu_x.shape, dtype=mu_x.dtype, device=mu_x.device)
        y_square = torch.matmul(factor.transpose(1, 2), y ** 2)
        y_mu_double = torch.matmul(2.0 * (factor * mu_x).transpose(1, 2), y)
        mu_square = torch.sum(factor * (mu_x ** 2), 1).unsqueeze(-1)
        log_prior = y_square - y_mu_double + mu_square + const
        attn = monotonic_align.maximum_path(log_prior, attn_mask.squeeze(1))
        return attn[0].sum(-1).round().long().cpu().numpy()

    # 合成 ------------------------------------------------------------
    @torch.inference_mode()
    def decode(self, mu_x, x_mask, spks, frames, gen_seed):
        """用指定的整數 frame 數合成,回傳 (decoder_outputs, attn)。與 synthesise() 相同的步驟。"""
        frames = torch.as_tensor(np.asarray(frames), dtype=mu_x.dtype, device=self.device)[None, None]
        y_len = int(frames.sum().item())
        y_max_ = fix_len_compatibility(y_len)
        y_lengths = torch.tensor([y_len], device=self.device)
        y_mask = sequence_mask(y_lengths, y_max_).unsqueeze(1).to(x_mask.dtype)
        attn_mask = x_mask.unsqueeze(-1) * y_mask.unsqueeze(2)
        attn = generate_path(frames.squeeze(1), attn_mask.squeeze(1)).unsqueeze(1)
        mu_y = torch.matmul(attn.squeeze(1).transpose(1, 2), mu_x.transpose(1, 2)).transpose(1, 2)
        torch.manual_seed(gen_seed)
        dec = self.model.decoder(mu_y, y_mask, C.N_TIMESTEPS, C.TEMPERATURE, spks)
        return dec[:, :, :y_len], attn[:, :, :, :y_len]

    def load_vocoder(self, path=None):
        from matcha.cli import load_hifigan
        from matcha.utils.utils import get_user_data_dir
        path = Path(path) if path else get_user_data_dir() / "hifigan_univ_v1"
        self.vocoder = load_hifigan(str(path), self.device)

    @torch.inference_mode()
    def to_wav(self, dec):
        if self.vocoder is None:
            self.load_vocoder()
        mel = dec * self.mel_std + self.mel_mean
        return self.vocoder(mel).clamp(-1, 1).squeeze().float().cpu().numpy()

    @torch.inference_mode()
    def selfcheck(self, text, spk_id, gen_seed=1234):
        """不改時長時,decode() 必須與 model.synthesise() 完全相同。"""
        from .durations import matcha_frames
        x, _ = self.text(text)
        mu_x, x_mask, spks, w = self.encode(x, spk_id)
        dec, _ = self.decode(mu_x, x_mask, spks, matcha_frames(w, mode="ceil"), gen_seed)
        torch.manual_seed(gen_seed)
        ref = self.model.synthesise(x, torch.tensor([x.shape[1]], device=self.device), C.N_TIMESTEPS,
                                    temperature=C.TEMPERATURE,
                                    spks=torch.tensor([spk_id], device=self.device) if self.model.n_spks > 1 else None,
                                    length_scale=1.0)["decoder_outputs"]
        if ref.shape != dec.shape:
            return False, f"shape {tuple(dec.shape)} vs {tuple(ref.shape)}"
        diff = (ref - dec).abs().max().item()
        return diff < 1e-5, f"max |diff| = {diff:.2e}"


def import_eval_v2():
    """沿用既有 evaluate_v2.py 的 MCD / F0 算法,確保與論文數字同一套程式。"""
    try:
        import evaluate_v2 as ev
    except Exception as e:  # noqa: BLE001
        raise ImportError(f"無法 import {C.ANALYSIS}/evaluate_v2.py:{e}") from e
    return ev
