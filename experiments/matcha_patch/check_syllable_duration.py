"""在真正的 Matcha-TTS(已套用 apply_patch.sh)上檢查音節雙層時長預測器:形狀、梯度、比例加總、損失。不需要 GPU。
    python check_syllable_duration.py            # 預設 ~/Matcha-TTS
"""
import sys, types, torch
sys.path.insert(0, sys.argv[1] if len(sys.argv) > 1 else __import__("os").path.expanduser("~/Matcha-TTS"))
from types import SimpleNamespace as NS
from matcha.text.symbols import HAKKA_SYMBOLS
from matcha.models.components.text_encoder import TextEncoder
from matcha.models.components.syllable_duration import syllable_losses
names = [None] + HAKKA_SYMBOLS
idx = {n: i for i, n in enumerate(names) if n}
def seq(toks):
    ids = []
    for t in toks: ids += [0, idx[t]]
    return ids + [0]
utts = [["s", "ii@5", "d", "d", "o@24", "ng", "sp", "n", "a@24", "n", "a@11"], ["h", "a@2", "b", "a@11"]]
xs = [seq(u) for u in utts]
L = max(map(len, xs)); x = torch.zeros(2, L, dtype=torch.long)
for b, s in enumerate(xs): x[b, :len(s)] = torch.tensor(s)
xl = torch.tensor([len(s) for s in xs])
enc_p = NS(n_feats=80, n_channels=192, filter_channels=768, n_heads=2, n_layers=6, kernel_size=3, p_dropout=0.1, prenet=True)
dp_p = NS(filter_channels_dp=512, kernel_size=5, p_dropout=0.1)
torch.manual_seed(0)
for mode in ("feat", "factor"):
    te = TextEncoder("RoPE Encoder", enc_p, dp_p, len(names), n_spks=4, spk_emb_dim=64, syl_dur=mode)
    spks = torch.randn(2, 64)
    mu, logw, xm = te(x, xl, spks)
    assert logw.shape == (2, 1, L) and torch.isfinite(logw).all()
    aux = te.proj_w.last_aux
    M = aux["M"]
    assert M.shape[1] == 4 and M[1].sum(-1).tolist()[:2] == [5.0, 1.0], M[1].sum(-1)   # h a@2 b + 2 個內部 blank;a@11 單獨
    # 目標時長:每個位置 1~6 個音框
    T = 60
    d = torch.randint(1, 6, (2, L)).float() * xm.squeeze(1)
    attn = torch.zeros(2, L, T)
    for b in range(2):
        t0 = 0
        for p in range(L):
            k = int(d[b, p]); attn[b, p, t0:t0 + k] = 1; t0 += k
    syl, share = syllable_losses(aux, attn, mode)
    assert torch.isfinite(syl) and (share is None) == (mode == "feat")
    loss = syl + (share if share is not None else 0) + ((logw - torch.log(d.clamp(min=1)).unsqueeze(1)) ** 2 * xm).mean()
    loss.backward()
    g = sum(p.grad.abs().sum() for p in te.proj_w.parameters() if p.grad is not None)
    assert g > 0
    if mode == "factor":
        w = torch.exp(logw).squeeze(1) * aux["in_syl"].squeeze(1)
        tot = torch.bmm(M, w.unsqueeze(-1)).squeeze(-1)
        assert torch.allclose(tot[aux["syl_mask"].squeeze(1) > 0], torch.exp(aux["s_log"].squeeze(1))[aux["syl_mask"].squeeze(1) > 0], rtol=1e-4)
    print(mode, "ok: syl_loss %.3f share_loss %s" % (syl.item(), "—" if share is None else "%.3f" % share.item()))
# 完全吻合時 KL = 0
aux = {"M": M, "syl_mask": aux["syl_mask"], "s_log": torch.log(torch.bmm(M, d.unsqueeze(-1)).squeeze(-1).clamp(min=1e-8)).unsqueeze(1) * aux["syl_mask"],
       "in_syl": aux["in_syl"]}
D_tok = torch.bmm(torch.bmm(M, d.unsqueeze(-1)).squeeze(-1).unsqueeze(1), M).squeeze(1)
aux["logp"] = (torch.log((d / D_tok.clamp(min=1e-8)).clamp(min=1e-8)) * aux["in_syl"].squeeze(1)).unsqueeze(1)
s, k = syllable_losses(aux, attn, "factor")
print("perfect:", round(s.item(), 6), round(k.item(), 6))
assert s.item() < 1e-5 and abs(k.item()) < 1e-5
# 查表:文字前端提供的分組優先於規則(把 "n a@24 n a@11" 的第二個 n 當成韻尾)
import json, tempfile
from matcha.models.components.hakka_syllables import table_key
seq_ids = seq(["n", "a@24", "n", "a@11"])
tf = tempfile.NamedTemporaryFile("w", suffix=".json", delete=False, encoding="utf-8")
json.dump({table_key(seq_ids): [0, 0, 0, 1]}, tf); tf.close()
te = TextEncoder("RoPE Encoder", enc_p, dp_p, len(names), n_spks=4, spk_emb_dim=64, syl_dur="factor", syl_table=tf.name)
xx = torch.tensor([seq_ids]); te(xx, torch.tensor([len(seq_ids)]), torch.randn(1, 64))
assert te.proj_w.last_aux["M"][0].sum(-1).tolist()[:2] == [5.0, 1.0] and te.proj_w.n_lookup == [1, 0]
print("查表 ok")
print("全部檢查通過")
