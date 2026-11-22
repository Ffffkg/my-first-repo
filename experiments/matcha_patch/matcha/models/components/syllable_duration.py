"""音節→音素雙層時長預測器(取代 TextEncoder 的 proj_w)。

第一層:把同一音節內各符號的編碼向量取平均,加上「是否在停頓/句末之前」,經兩層卷積(沿音節序列)
        預測每個音節的對數總長(音框)。
第二層:把預測的音節總長(不回傳梯度)當作額外特徵提供給音節內每個符號,用原本結構的時長預測器輸出每個符號的分數 r。
  mode="feat"   (Moungsri 等人的做法):logw = r,音節總長只當作提示特徵。
  mode="factor" (音節總長 × 音節內比例):音節內符號 logw = 音節對數總長 + log softmax(音節內的 r);
                 不屬於音節的位置(停頓、音節之間的 blank)logw = r。
推論介面不變:輸出每個符號的 logw,後續取整、合成流程與原本相同。
訓練損失由 MatchaTTS.forward 加上(見 last_aux):音節總長的對數 L1,以及 factor 模式的音節內比例 KL。
"""
import torch
from torch import nn

from matcha.models.components.hakka_syllables import segment
from matcha.models.components.text_encoder import DurationPredictor, LayerNorm


class SyllableDurationPredictor(nn.Module):
    def __init__(self, in_channels, filter_channels, kernel_size, p_dropout, mode, symbol_names,
                 nasal_rule="onset", feat_dim=16, pp_dim=8):
        super().__init__()
        assert mode in ("feat", "factor"), mode
        self.mode = mode
        self.names = list(symbol_names)            # names[id];不存進 state_dict
        self.nasal_rule = nasal_rule
        self.pp_emb = nn.Embedding(2, pp_dim)
        self.syl_conv1 = nn.Conv1d(in_channels + pp_dim, filter_channels, 3, padding=1)
        self.syl_norm1 = LayerNorm(filter_channels)
        self.syl_conv2 = nn.Conv1d(filter_channels, filter_channels, 3, padding=1)
        self.syl_norm2 = LayerNorm(filter_channels)
        self.syl_proj = nn.Conv1d(filter_channels, 1, 1)
        self.drop = nn.Dropout(p_dropout)
        self.feat = nn.Conv1d(2, feat_dim, 1)
        self.tok_dp = DurationPredictor(in_channels + feat_dim, filter_channels, kernel_size, p_dropout)
        self.last_aux = None

    def _segments(self, x_ids, x_lengths, device):
        B, L = x_ids.shape
        ids = x_ids.detach().cpu().tolist()
        lens = x_lengths.detach().cpu().tolist()
        segs = [segment(ids[b][: int(lens[b])], self.names, self.nasal_rule) for b in range(B)]
        S = max(1, max(len(pp) for _, pp in segs))
        M = torch.zeros(B, S, L, device=device)
        pp = torch.zeros(B, S, dtype=torch.long, device=device)
        for b, (syl, ppb) in enumerate(segs):
            pos = [p for p, s in enumerate(syl) if s >= 0]
            if pos:
                M[b, torch.tensor([syl[p] for p in pos], device=device), torch.tensor(pos, device=device)] = 1.0
            if ppb:
                pp[b, : len(ppb)] = torch.tensor(ppb, dtype=torch.long, device=device)
        return M, pp

    def forward(self, x, x_mask, x_ids, x_lengths):
        M, pp = self._segments(x_ids, x_lengths, x.device)          # M: (B, S, L)
        cnt = M.sum(-1)                                              # (B, S)
        syl_mask = (cnt > 0).to(x.dtype).unsqueeze(1)                # (B, 1, S)
        pooled = torch.bmm(M, x.transpose(1, 2)) / cnt.clamp(min=1).unsqueeze(-1)    # (B, S, C)
        h = torch.cat([pooled.transpose(1, 2), self.pp_emb(pp).transpose(1, 2)], dim=1)
        h = self.drop(self.syl_norm1(torch.relu(self.syl_conv1(h * syl_mask))))
        h = self.drop(self.syl_norm2(torch.relu(self.syl_conv2(h * syl_mask))))
        s_log = self.syl_proj(h * syl_mask) * syl_mask               # (B, 1, S) 音節對數總長(音框)

        in_syl = M.sum(1, keepdim=True)                              # (B, 1, L),0/1
        s_tok = torch.bmm(s_log, M)                                  # (B, 1, L)
        f = self.feat(torch.cat([s_tok.detach() * in_syl, in_syl], dim=1)) * x_mask
        r = self.tok_dp(torch.cat([x, f], dim=1), x_mask)            # (B, 1, L)

        logp = None
        if self.mode == "feat":
            logw = r
        else:
            R = r.expand(-1, M.shape[1], -1).masked_fill(M == 0, -1e4)       # (B, S, L)
            lse = torch.logsumexp(R, dim=-1) * syl_mask.squeeze(1)            # (B, S)
            lse_tok = torch.bmm(lse.unsqueeze(1), M)                          # (B, 1, L)
            logp = (r - lse_tok) * in_syl
            logw = torch.where(in_syl > 0, s_tok + logp, r)
        self.last_aux = {"M": M, "syl_mask": syl_mask, "s_log": s_log, "in_syl": in_syl, "logp": logp}
        return logw * x_mask


def syllable_losses(aux, attn, mode):
    """attn: (B, L, T) 訓練用的對齊路徑。回傳 (音節總長對數 L1, 音節內比例 KL 或 None)。"""
    M = aux["M"]
    d = attn.sum(-1)                                                 # (B, L) 每個符號的目標音框數
    D = torch.bmm(M, d.unsqueeze(-1)).squeeze(-1)                    # (B, S) 音節總長目標
    valid = (aux["syl_mask"].squeeze(1) > 0) & (D > 0)
    n = valid.sum().clamp(min=1)
    s_log = aux["s_log"].squeeze(1)
    syl = (torch.abs(s_log - torch.log(D.clamp(min=1e-8))) * valid).sum() / n
    if mode != "factor":
        return syl, None
    D_tok = torch.bmm(D.unsqueeze(1), M).squeeze(1)                  # (B, L)
    in_syl = aux["in_syl"].squeeze(1) > 0
    q = torch.where(in_syl & (D_tok > 0), d / D_tok.clamp(min=1e-8), torch.zeros_like(d))
    logp = aux["logp"].squeeze(1)
    kl = torch.where(q > 0, q * (torch.log(q.clamp(min=1e-8)) - logp), torch.zeros_like(q))
    return syl, kl.sum() / n
