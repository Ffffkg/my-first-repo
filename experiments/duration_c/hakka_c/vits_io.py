"""VITS 版的 Engine:介面與 matcha_io.Engine 相同(text / encode / decode / to_wav / wav),
所以 c_eval.py、c3_target_check.py、d1_pipeline.sh 只要加 --arch vits 就能評估 VITS 模型。

差別:VITS 直接輸出波形,decode() 回傳的就是波形,to_wav() 只是轉成 numpy;不需要 HiFi-GAN 聲碼器與 mel 統計量。
"""
import re
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

from . import config as C
from . import vits_model as V

NOISE_SCALE = 0.667        # 官方推論預設(與 Matcha 的 temperature 0.667 相同)


def find_vits_ckpt(exp, which="last"):
    d = C.VITS_RUNS / exp
    gs = {int(m.group(1)): p for p in d.glob("G_*.pth") if (m := re.fullmatch(r"G_(\d+)\.pth", p.name))}
    if not gs:
        raise FileNotFoundError(f"{d} 沒有 G_*.pth")
    if which in ("last", "best"):
        return gs[max(gs)]
    m = re.fullmatch(r"step(\d+)", which)
    if m:
        n = int(m.group(1))
        if n not in gs:
            raise FileNotFoundError(f"{d} 沒有第 {n} 步;現有 {sorted(gs)}")
        return gs[n]
    return Path(which)


class VitsEngine:
    def __init__(self, ckpt, device="cuda", cleaners=("hakka_cleaners",)):
        self.device = torch.device(device if torch.cuda.is_available() or device == "cpu" else "cpu")
        st = torch.load(str(ckpt), map_location=self.device, weights_only=False)
        a = st["config"]["args"]
        self.model, _ = V.build(st["config"]["n_vocab"], st["config"]["n_speakers"], a["dp"], a["dp_kernel"], a["dp_filter"])
        self.model.load_state_dict(st["G"])
        self.model.to(self.device).eval()
        self.cleaners = list(cleaners)
        from .pua import resolve_pua_map
        self.pua, src = resolve_pua_map(self.cleaners, verbose=False)
        print(f"PUA 對照表:{src};VITS 第 {st['step']} 步")
        self.ckpt = ckpt

    def text(self, text):
        sys.path.insert(0, str(C.MATCHA))
        from matcha.text import text_to_sequence
        from matcha.utils.utils import intersperse
        seq, clean = text_to_sequence(text, self.cleaners)
        names = []
        for ch in clean:
            if ch not in self.pua:
                raise KeyError(f"PUA 字元 U+{ord(ch):04X} 不在對照表中")
            names.append(self.pua[ch])
        return torch.tensor(intersperse(seq, 0), dtype=torch.long, device=self.device)[None], names

    @torch.inference_mode()
    def encode(self, x, spk_id):
        """回傳 (狀態, x_mask, 語者條件, w);狀態只給 decode() 用。"""
        sid = torch.tensor([spk_id], device=self.device)
        m_p, logs_p, x_mask, g, w = self.model.encode(x, torch.tensor([x.shape[1]], device=self.device), sid)
        return (m_p, logs_p), x_mask, g, w[0, 0].float().cpu().numpy()

    def wav(self, path):
        a, sr = sf.read(str(path), dtype="float32", always_2d=True)
        assert sr == C.SR, f"{path} 取樣率 {sr} != {C.SR}"
        return a.mean(axis=1)

    @torch.inference_mode()
    def decode(self, state, x_mask, g, frames, gen_seed):
        m_p, logs_p = state
        fr = torch.as_tensor(np.asarray(frames), dtype=m_p.dtype, device=self.device)[None, None]
        torch.manual_seed(gen_seed)
        o, attn = self.model.decode(m_p, logs_p, x_mask, g, fr, NOISE_SCALE)
        return o, attn

    def load_vocoder(self, path=None):
        pass                                  # VITS 自己輸出波形

    @torch.inference_mode()
    def to_wav(self, o):
        return o.clamp(-1, 1).squeeze().float().cpu().numpy()

    @torch.inference_mode()
    def selfcheck(self, text, spk_id, gen_seed=1234):
        """不改時長時,encode() + decode() 必須與官方 infer() 逐點相同。"""
        from .durations import matcha_frames
        x, _ = self.text(text)
        state, x_mask, g, w = self.encode(x, spk_id)
        o, _ = self.decode(state, x_mask, g, matcha_frames(w, mode="ceil"), gen_seed)
        torch.manual_seed(gen_seed)
        ref, *_ = self.model.infer(x, torch.tensor([x.shape[1]], device=self.device),
                                   sid=torch.tensor([spk_id], device=self.device), noise_scale=NOISE_SCALE)
        if ref.shape != o.shape:
            return False, f"shape {tuple(o.shape)} vs {tuple(ref.shape)}"
        diff = (ref - o).abs().max().item()
        return diff < 1e-4, f"max |diff| = {diff:.2e}"
