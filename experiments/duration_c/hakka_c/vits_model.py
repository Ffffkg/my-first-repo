"""VITS(Kim et al., ICML 2021,官方程式 jaywalnut310/vits)的客語版本:不改官方原始碼,只在這裡包一層。

和官方的差別只有四點,其餘(TextEncoder、PosteriorEncoder、flow、HiFi-GAN 解碼器、判別器、損失)都直接用官方模組:
  1. 文字前端用 Matcha 的 hakka_cleaners 與符號表(與 Matcha 系統同一套音素與 blank 穿插),時長檔可直接共用。
  2. 時長預測器預設用確定性 DurationPredictor(官方 use_sdp=False 的選項),kernel / 通道數可調,
     與 Matcha 的 DP 同構(預設 kernel 3、256 通道;XAD 的「放大 DP」= kernel 5、512 通道)。
  3. forward(dur=...) 給了 MFA 時長時,用它取代 MAS 的對齊路徑(= +AD);沒給就照官方跑 MAS。
  4. MAS 改用 Matcha 已編譯好的 monotonic_align(官方版要另外編 Cython);兩者演算法相同,只差矩陣轉置。
頻譜改用 torch.stft(return_complex=True)與關鍵字參數的 librosa mel,其餘數值與官方 mel_processing.py 相同。
"""
import math
import sys
import types

import torch
from torch.nn import functional as F

from . import config as C

VITS_COMMIT = "2e561ba"   # 測試過的官方版本(v_setup.sh 會 checkout 這個 commit)


def _mas(neg_cent, mask):
    """官方 maximum_path 的輸入是 [b, t_y, t_x];Matcha(Glow-TTS)的是 [b, t_x, t_y]。"""
    from matcha.utils.monotonic_align import maximum_path
    return maximum_path(neg_cent.transpose(1, 2).contiguous(), mask.transpose(1, 2).contiguous()).transpose(1, 2)


def import_vits():
    """把官方程式資料夾加進 sys.path 並匯入;回傳 (models, commons, losses)。"""
    root = C.VITS
    if not (root / "models.py").exists():
        raise FileNotFoundError(f"找不到官方 VITS 程式:{root}/models.py(先跑 v_setup.sh,或設定 VITS_ROOT)")
    for p in (C.MATCHA, root):
        if str(p) not in sys.path:
            sys.path.insert(0, str(p))
    if "monotonic_align" not in sys.modules:          # 官方 models.py 在最上面 import monotonic_align
        shim = types.ModuleType("monotonic_align")
        shim.maximum_path = _mas
        sys.modules["monotonic_align"] = shim
    import commons
    import losses
    import models
    for m in (commons, losses, models):                # 避免撞名(例如其他資料夾也有 models.py)
        if not str(getattr(m, "__file__", "")).startswith(str(root)):
            raise ImportError(f"{m.__name__} 不是從 {root} 匯入的:{m.__file__}")
    return models, commons, losses


# ---------------------------------------------------------------- 頻譜(與官方 mel_processing.py 相同的數值)
_cache = {}


def _window(win, ref):
    k = ("win", win, ref.dtype, ref.device)
    if k not in _cache:
        _cache[k] = torch.hann_window(win, dtype=ref.dtype, device=ref.device)
    return _cache[k]


def spectrogram(y, n_fft=C.N_FFT, hop=C.HOP, win=C.WIN):
    """y: [b, L] → 線性幅度譜 [b, n_fft//2+1, L//hop]"""
    p = (n_fft - hop) // 2
    y = F.pad(y.unsqueeze(1), (p, p), mode="reflect").squeeze(1)
    spec = torch.stft(y, n_fft, hop_length=hop, win_length=win, window=_window(win, y), center=False,
                      normalized=False, onesided=True, return_complex=True)
    return torch.sqrt(spec.real ** 2 + spec.imag ** 2 + 1e-6)


def spec_to_mel(spec, n_fft=C.N_FFT, n_mels=C.N_MELS, sr=C.SR, fmin=0.0, fmax=None):
    k = ("mel", n_fft, n_mels, sr, fmin, fmax, spec.dtype, spec.device)
    if k not in _cache:
        from librosa.filters import mel as librosa_mel
        _cache[k] = torch.from_numpy(librosa_mel(sr=sr, n_fft=n_fft, n_mels=n_mels, fmin=fmin, fmax=fmax)).to(spec)
    return torch.log(torch.clamp(torch.matmul(_cache[k], spec), min=1e-5))


def mel_from_wav(y):
    return spec_to_mel(spectrogram(y))


# ---------------------------------------------------------------- 模型
MODEL_HPS = {   # 官方 configs/vctk_base.json 的 model 區塊(多語者)
    "inter_channels": 192, "hidden_channels": 192, "filter_channels": 768, "n_heads": 2, "n_layers": 6,
    "kernel_size": 3, "p_dropout": 0.1, "resblock": "1", "resblock_kernel_sizes": [3, 7, 11],
    "resblock_dilation_sizes": [[1, 3, 5], [1, 3, 5], [1, 3, 5]], "upsample_rates": [8, 8, 2, 2],
    "upsample_initial_channel": 512, "upsample_kernel_sizes": [16, 16, 4, 4], "gin_channels": 256,
}
TRAIN_HPS = {   # 官方 configs/vctk_base.json 的 train 區塊
    "learning_rate": 2e-4, "betas": [0.8, 0.99], "eps": 1e-9, "lr_decay": 0.999875,
    "segment_size": 8192, "c_mel": 45, "c_kl": 1.0,
}


def build(n_vocab, n_speakers, dp="det", dp_kernel=3, dp_filter=256):
    models, commons, _ = import_vits()

    class HakkaVits(models.SynthesizerTrn):
        def __init__(self):
            super().__init__(n_vocab, C.N_FFT // 2 + 1, TRAIN_HPS["segment_size"] // C.HOP,
                             n_speakers=n_speakers, use_sdp=(dp == "sdp"), **MODEL_HPS)
            if dp == "det":
                self.dp = models.DurationPredictor(MODEL_HPS["hidden_channels"], dp_filter, dp_kernel, 0.5,
                                                   gin_channels=MODEL_HPS["gin_channels"])

        def forward(self, x, x_lengths, y, y_lengths, sid=None, dur=None):
            """與官方 forward 相同;dur([b, t_x] 整數 frame)不為 None 時用它取代 MAS 的路徑。"""
            x, m_p, logs_p, x_mask = self.enc_p(x, x_lengths)
            g = self.emb_g(sid).unsqueeze(-1) if self.n_speakers > 0 else None
            z, m_q, logs_q, y_mask = self.enc_q(y, y_lengths, g=g)
            z_p = self.flow(z, y_mask, g=g)
            attn_mask = torch.unsqueeze(x_mask, 2) * torch.unsqueeze(y_mask, -1)       # [b, 1, t_y, t_x]
            if dur is None:
                with torch.no_grad():
                    s_p_sq_r = torch.exp(-2 * logs_p)
                    neg_cent = (torch.sum(-0.5 * math.log(2 * math.pi) - logs_p, [1], keepdim=True)
                                + torch.matmul(-0.5 * (z_p ** 2).transpose(1, 2), s_p_sq_r)
                                + torch.matmul(z_p.transpose(1, 2), (m_p * s_p_sq_r))
                                + torch.sum(-0.5 * (m_p ** 2) * s_p_sq_r, [1], keepdim=True))
                    attn = _mas(neg_cent.float(), attn_mask.squeeze(1).float()).unsqueeze(1).to(m_p.dtype).detach()
            else:
                attn = commons.generate_path(dur.unsqueeze(1).float(), attn_mask.float()).to(m_p.dtype).detach()
            w = attn.sum(2)
            if self.use_sdp:
                l_length = self.dp(x, x_mask, w, g=g) / torch.sum(x_mask)
            else:
                logw_ = torch.log(w + 1e-6) * x_mask
                logw = self.dp(x, x_mask, g=g)
                l_length = torch.sum((logw - logw_) ** 2, [1, 2]) / torch.sum(x_mask)
            m_p = torch.matmul(attn.squeeze(1), m_p.transpose(1, 2)).transpose(1, 2)
            logs_p = torch.matmul(attn.squeeze(1), logs_p.transpose(1, 2)).transpose(1, 2)
            z_slice, ids_slice = commons.rand_slice_segments(z, y_lengths, self.segment_size)
            o = self.dec(z_slice, g=g)
            return o, l_length, attn, ids_slice, x_mask, y_mask, (z, z_p, m_p, logs_p, m_q, logs_q)

        # ---- 推論(拆成兩段,讓評估程式可以換時長;合起來與官方 infer() 逐點相同)
        def encode(self, x, x_lengths, sid=None, noise_scale_w=0.8):
            x, m_p, logs_p, x_mask = self.enc_p(x, x_lengths)
            g = self.emb_g(sid).unsqueeze(-1) if self.n_speakers > 0 else None
            if self.use_sdp:
                logw = self.dp(x, x_mask, g=g, reverse=True, noise_scale=noise_scale_w)
            else:
                logw = self.dp(x, x_mask, g=g)
            return m_p, logs_p, x_mask, g, torch.exp(logw) * x_mask

        def decode(self, m_p, logs_p, x_mask, g, frames, noise_scale=0.667):
            """frames: [b, 1, t_x] 每個符號的整數 frame 數"""
            y_lengths = torch.clamp_min(torch.sum(frames, [1, 2]), 1).long()
            y_mask = torch.unsqueeze(commons.sequence_mask(y_lengths, None), 1).to(x_mask.dtype)
            attn_mask = torch.unsqueeze(x_mask, 2) * torch.unsqueeze(y_mask, -1)
            attn = commons.generate_path(frames, attn_mask)
            m = torch.matmul(attn.squeeze(1), m_p.transpose(1, 2)).transpose(1, 2)
            logs = torch.matmul(attn.squeeze(1), logs_p.transpose(1, 2)).transpose(1, 2)
            z_p = m + torch.randn_like(m) * torch.exp(logs) * noise_scale
            z = self.flow(z_p, y_mask, g=g, reverse=True)
            return self.dec(z * y_mask, g=g), attn

    return HakkaVits(), models.MultiPeriodDiscriminator(False)


def n_vocab():
    if str(C.MATCHA) not in sys.path:
        sys.path.insert(0, str(C.MATCHA))
    from matcha.text.symbols import symbols
    return len(symbols)


def load_transfer(net, state, tag):
    """跨腔初始化:形狀相同的參數全部載入;形狀不同(例如放大的 DP)保留隨機初始化並列出。"""
    own = net.state_dict()
    ok = {k: v for k, v in state.items() if k in own and own[k].shape == v.shape}
    skipped = sorted(k for k in state if k in own and own[k].shape != state[k].shape)
    missing = sorted(k for k in own if k not in state)
    own.update(ok)
    net.load_state_dict(own)
    print(f"[transfer] {tag}:載入 {len(ok)}/{len(own)} 個參數;形狀不同而重新初始化 {len(skipped)} 個"
          f"{'(' + ', '.join(sorted({k.split('.')[0] for k in skipped})) + ')' if skipped else ''};"
          f"來源缺少 {len(missing)} 個")
    return skipped, missing
