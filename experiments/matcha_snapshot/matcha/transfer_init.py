#!/usr/bin/env python3
"""
transfer_init.py — 把來源腔 checkpoint 的相容權重載入目標模型

用於跨腔遷移:只複製形狀相容的權重(音素/聲調 embedding、編碼器、解碼器),
speaker embedding 因語者不同而重置。從 step 0 開始微調,非續訓。

被 train.py 在建立模型後、訓練開始前呼叫(見 patch)。
"""
import torch


def transfer_weights(model, src_ckpt, verbose=True):
    sd_src = torch.load(src_ckpt, map_location="cpu",
                        weights_only=False)["state_dict"]
    sd_tgt = model.state_dict()
    copied, skipped = [], []
    new_sd = {}
    for k, v in sd_tgt.items():
        # speaker embedding 語義綁定語者身分,跨腔不可沿用 → 保留目標初始化
        is_spk = ("spk" in k.lower()) or ("speaker" in k.lower())
        if (k in sd_src and sd_src[k].shape == v.shape and not is_spk):
            new_sd[k] = sd_src[k]
            copied.append(k)
        else:
            new_sd[k] = v          # spk_emb / 形狀不符 → 保留目標初始化
            skipped.append(k)
    model.load_state_dict(new_sd)
    if verbose:
        print(f"[transfer] 複製 {len(copied)} 個張量,"
              f"重置 {len(skipped)} 個(語者相關/形狀不符)")
        # 印出被跳過的關鍵層,方便確認
        key_skip = [k for k in skipped if "spk" in k.lower()
                    or "emb" in k.lower()][:6]
        if key_skip:
            print(f"[transfer] 重置範例:{key_skip}")
    return model
