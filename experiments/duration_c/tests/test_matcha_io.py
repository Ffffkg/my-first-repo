"""驗證 hakka_c.matcha_io 的 MAS 與合成路徑和 Matcha 原始程式完全一致。

需要 torch 與 Matcha-TTS(MATCHA_ROOT);沒有就自動跳過。用隨機初始化的小模型,不需要 checkpoint。
在你的 tts 環境跑,等於直接檢查「你本機(可能 patch 過)的 Matcha」:
    python -m pytest tests/test_matcha_io.py -q
"""
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

torch = pytest.importorskip("torch")
mio = pytest.importorskip("hakka_c.matcha_io")

from hakka_c import config as C  # noqa: E402
from hakka_c.durations import matcha_frames  # noqa: E402


@pytest.fixture(scope="module")
def engine():
    from omegaconf import OmegaConf
    from matcha.models.matcha_tts import MatchaTTS
    cfg_dir = C.MATCHA / "configs" / "model"
    enc = OmegaConf.load(cfg_dir / "encoder" / "default.yaml")
    dec = OmegaConf.load(cfg_dir / "decoder" / "default.yaml")
    cfm = OmegaConf.load(cfg_dir / "cfm" / "default.yaml")
    for c in (enc, dec, cfm):
        if "defaults" in c:
            del c["defaults"]
    enc.encoder_params.n_feats = 80                     # 原檔是 ${model.n_feats} 這類插值,單獨載入時要填實值
    enc.encoder_params.n_spks = 2
    enc.encoder_params.spk_emb_dim = 64
    enc.duration_predictor_params.filter_channels_dp = enc.encoder_params.filter_channels_dp
    enc.duration_predictor_params.p_dropout = enc.encoder_params.p_dropout
    torch.manual_seed(0)
    model = MatchaTTS(n_vocab=12, n_spks=2, spk_emb_dim=64, n_feats=80, encoder=enc, decoder=dec, cfm=cfm,
                      data_statistics={"mel_mean": -5.5, "mel_std": 2.1}, out_size=None)
    return mio.Engine(None, device="cpu", model=model)


def _x(n=9, seed=0):
    g = torch.Generator().manual_seed(seed)
    ids = torch.randint(1, 12, (n,), generator=g).tolist()
    from matcha.utils.utils import intersperse
    return torch.tensor(intersperse(ids, 0))[None]


def test_decode_equals_synthesise(engine):
    for seed in range(3):
        x = _x(7 + seed, seed)
        mu_x, x_mask, spks, w = engine.encode(x, 1)
        dec, _ = engine.decode(mu_x, x_mask, spks, matcha_frames(w, mode="ceil"), gen_seed=1234 + seed)
        torch.manual_seed(1234 + seed)
        ref = engine.model.synthesise(x, torch.tensor([x.shape[1]]), C.N_TIMESTEPS, temperature=C.TEMPERATURE,
                                      spks=torch.tensor([1]), length_scale=1.0)
        assert ref["decoder_outputs"].shape == dec.shape
        assert torch.allclose(ref["decoder_outputs"], dec, atol=1e-5)
        # attn 推得的時長也要等於 matcha_frames(ceil)
        assert ref["attn"][0, 0].sum(-1).long().tolist() == matcha_frames(w, mode="ceil").tolist()


def test_mas_equals_forward(engine):
    import inspect
    from matcha.models.matcha_tts import MatchaTTS
    if "return dur_loss, prior_loss, diff_loss, attn" not in inspect.getsource(MatchaTTS.forward):
        pytest.skip("這個 Matcha 版本的 forward 不回傳 attn")
    x = _x(10, 5)
    T = 160
    y = torch.randn(1, 80, T, generator=torch.Generator().manual_seed(1))
    mu_x, x_mask, _, _ = engine.encode(x, 0)
    mine = engine.mas(mu_x, x_mask, y)
    assert mine.sum() == T and mine.min() >= 1
    with torch.no_grad():
        _, _, _, attn = engine.model(x, torch.tensor([x.shape[1]]), y, torch.tensor([T]), spks=torch.tensor([0]))
    assert attn[0].sum(-1).long().tolist() == mine.tolist()


def test_mel_length_is_samples_div_hop():
    from matcha.utils.audio import mel_spectrogram
    for n in (22050, 22050 + 100, 54321):
        a = torch.rand(1, n) * 0.2 - 0.1
        m = mel_spectrogram(a, C.N_FFT, C.N_MELS, C.SR, C.HOP, C.WIN, C.F_MIN, C.F_MAX, center=False)
        assert m.shape[-1] == n // C.HOP


def test_precomputed_durations_path(engine):
    """C2:給 durations 時,forward 用的對齊與 durations 完全相同。"""
    import inspect
    from matcha.models.matcha_tts import MatchaTTS
    if "use_precomputed_durations" not in inspect.getsource(MatchaTTS):
        pytest.skip("舊版 Matcha 沒有 use_precomputed_durations")
    from hakka_c.durations import mfa_to_frames
    n = 6
    x = _x(n, 3)
    T = 120
    iv = [(0.05 + 0.2 * k, 0.05 + 0.2 * k + 0.15) for k in range(n)]
    fr = mfa_to_frames(n, list(range(n)), iv, T)
    m = engine.model
    m.use_precomputed_durations = True
    try:
        y = torch.randn(1, 80, T)
        with torch.no_grad():
            dur_loss, _, _, attn = m(x, torch.tensor([x.shape[1]]), y, torch.tensor([T]), spks=torch.tensor([0]),
                                     durations=torch.tensor(fr)[None, None])
        assert attn[0].sum(-1).long().tolist() == fr.tolist()
        assert np.isfinite(dur_loss.item())
    finally:
        m.use_precomputed_durations = False
