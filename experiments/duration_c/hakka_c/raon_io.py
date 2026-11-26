"""第三種架構:Raon-OpenTTS(KRAFTON 開源的擴散 Transformer + 流匹配 TTS,沿用 F5-TTS 架構)。

DiTTo-TTS(同一團隊,ICLR 2025)沒有公開程式碼與權重;Raon-OpenTTS 是同團隊之後完整開源(程式碼、權重、資料)
的版本,和 DiTTo-TTS 一樣「沒有逐音素時長預測器,只給整句總長」,模型自己在內部決定每個音的長短。

這個檔案只放純函式與模型載入,不碰 Matcha。需要 raon 環境(r_setup.sh 建立)才能 import torch / f5_tts 的部分;
純函式(文字轉換、語速擬合、批次切分、詞表擴充)不需要 GPU,tests/test_pure.py 有測。
"""
import json
import os
import random
from pathlib import Path

import numpy as np

from . import config as C

RAON = Path(os.environ.get("RAON_ROOT", C.HAKKA / "raon")).expanduser()          # 資料、checkpoint、預訓練檔
RAON_REPO = Path(os.environ.get("RAON_REPO", "~/Raon-OpenTTS")).expanduser()       # 官方程式碼(r_setup.sh 下載)
PRETRAINED = RAON / "pretrained" / "Raon-OpenTTS-0.3B"
HIFIGAN16K = RAON / "pretrained" / "hifigan16k" / "generator.ckpt"
SR16 = 16000
HOP16 = 256
NFFT16 = 1024
FPS16 = SR16 / HOP16                    # 62.5 音框/秒(1 個音框 16 ms)
# 官方推論預設(infer/utils_infer.py)
NFE_STEPS = 32
CFG_STRENGTH = 2.0
SWAY = -1.0
TARGET_RMS = 0.1


# ---------------------------------------------------------------- 純函式
def mel_frames(n_samples):
    """sbhifigan16k 的 mel 長度:center=False,裁成 hop 對齊 → 1 + (L − n_fft) // hop。"""
    return max(1, 1 + (int(n_samples) - NFFT16) // HOP16)


def words_text(words):
    """TextGrid words 層的拼音音節(已去掉靜音)→ 模型輸入文字:音節之間一個空白。"""
    return " ".join(w.strip() for w in words if w.strip())


def fit_rate(rows):
    """每位語者「整句秒數 = a + b × 字元數」的最小平方解。rows: [{spk, sec, text}]。
    推論時只給文字長度就能估整句總長(模型本身沒有時長預測器)。回傳 {spk: (a, b)}。"""
    by = {}
    for r in rows:
        by.setdefault(int(r["spk"]), []).append((len(r["text"]), float(r["sec"])))
    out = {}
    for spk, xs in sorted(by.items()):
        n = np.array([x for x, _ in xs], float)
        s = np.array([y for _, y in xs], float)
        if len(xs) < 2 or np.ptp(n) == 0:
            out[spk] = (0.0, float(s.sum() / max(n.sum(), 1)))
            continue
        b, a = np.polyfit(n, s, 1)
        out[spk] = (float(a), float(b))
    return out


def rate_seconds(rate, spk, text):
    a, b = rate[int(spk)] if int(spk) in rate else rate[str(spk)]
    return max(0.3, a + b * len(text))


def pick_prompt(rows, lo=3.0, hi=6.0, target=4.5):
    """每位語者挑一句訓練句當聲音提示:長度在 [lo, hi] 秒、最接近 target;同分取檔名較小者。"""
    out = {}
    for r in sorted(rows, key=lambda r: r["stem"]):
        if not lo <= r["sec"] <= hi:
            continue
        spk = int(r["spk"])
        if spk not in out or abs(r["sec"] - target) < abs(out[spk]["sec"] - target):
            out[spk] = r
    return out


def extend_vocab(old_map, texts):
    """old_map: {字元: 編號}(官方 vocab.txt)。缺少的字元依序接在後面。回傳 (新 map, 新增字元清單)。"""
    if old_map and max(old_map.values()) != len(old_map) - 1:
        raise ValueError("vocab.txt 有重複的字元,編號不連續")
    need = sorted({ch for t in texts for ch in t})
    new = [ch for ch in need if ch not in old_map]
    m = dict(old_map)
    for k, ch in enumerate(new):
        m[ch] = len(old_map) + k
    return m, new


def read_vocab(path):
    """與 f5_tts.model.utils.get_tokenizer(…, "custom") 完全相同的解析方式(每行去掉最後一個字元)。"""
    m = {}
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            m[line[:-1]] = i
    return m


def write_vocab(m, path):
    toks = sorted(m, key=m.get)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for t in toks:
            f.write(t + "\n")


def frame_batches(lengths, max_frames, seed, max_items=64):
    """依長度分批:同一批的「最長 × 句數」≤ max_frames(含 padding)。
    先依長度加一點隨機擾動排序,切好後再打亂批次順序。回傳索引清單的清單。"""
    rng = random.Random(seed)
    order = sorted(range(len(lengths)), key=lambda i: lengths[i] * (1 + 0.1 * rng.random()))
    batches, cur, mx = [], [], 0
    for i in order:
        m2 = max(mx, lengths[i])
        if cur and (m2 * (len(cur) + 1) > max_frames or len(cur) >= max_items):
            batches.append(cur)
            cur, m2 = [], lengths[i]
        cur.append(i)
        mx = m2
    if cur:
        batches.append(cur)
    rng.shuffle(batches)
    return batches


def lr_at(step, peak, warmup, total):
    """線性暖身到 peak,之後線性降到 0(與官方訓練的排程同形)。step 從 1 起算。"""
    if step <= warmup:
        return peak * step / max(warmup, 1)
    return peak * max(0.0, (total - step) / max(total - warmup, 1))


def data_dir(dialect):
    return RAON / "data" / dialect


def load_jsonl(path):
    return [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]


def exp_dir(exp):
    return RAON / "runs" / exp


def find_raon_ckpt(exp, which="last"):
    d = exp_dir(exp)
    cks = {int(p.stem.split("_")[1]): p for p in d.glob("model_*.pt") if p.stem.split("_")[1].isdigit()}
    if not cks:
        raise FileNotFoundError(f"{d} 沒有 model_<步數>.pt")
    if which in ("last", "best"):
        return cks[max(cks)]
    if which.startswith("step"):
        n = int(which[4:])
        if n not in cks:
            raise FileNotFoundError(f"{d} 沒有第 {n} 步;現有 {sorted(cks)}")
        return cks[n]
    return Path(which)


# ---------------------------------------------------------------- 預訓練檔
def find_pretrained(root=PRETRAINED):
    """回傳 (checkpoint, vocab.txt)。checkpoint 優先 model_last.pt,其次 .safetensors,再其次最大的 .pt。"""
    root = Path(root)
    files = [p for p in root.rglob("*") if p.is_file() and ".cache" not in p.parts]
    if not files:
        raise FileNotFoundError(f"{root} 是空的:先跑 bash r_setup.sh")
    ck = [p for p in files if p.name == "model_last.pt"] or sorted(p for p in files if p.suffix == ".safetensors") \
        or sorted((p for p in files if p.suffix in (".pt", ".pth")), key=lambda p: -p.stat().st_size)
    vocab = [p for p in files if p.name == "vocab.txt"] or sorted(p for p in files if "vocab" in p.name and p.suffix == ".txt") \
        or [p for p in [RAON_REPO / "src" / "f5_tts" / "data" / "vocab.txt"] if p.exists()]
    if not ck or not vocab:
        raise FileNotFoundError(f"{root} 找不到 checkpoint 或 vocab.txt;現有檔案:{[str(p.relative_to(root)) for p in files]}")
    return ck[0], vocab[0]


def model_cfg():
    """官方 0.3B 設定(configs/03b.yaml 的 model.arch 與 model.mel_spec)。"""
    import yaml
    y = yaml.safe_load(open(RAON_REPO / "src" / "f5_tts" / "configs" / "03b.yaml", encoding="utf-8"))
    return dict(y["model"]["arch"]), dict(y["model"]["mel_spec"])


def clean_state(sd):
    """官方 checkpoint 可能是 {ema_model_state_dict}、{model_state_dict} 或直接是權重(safetensors)。
    一律轉成 CFM 的 state_dict 鍵名(去掉 ema_model. 前綴與 EMA 計數、舊版 mel 緩衝)。"""
    if "ema_model_state_dict" in sd:
        sd = sd["ema_model_state_dict"]
    elif "model_state_dict" in sd:
        sd = sd["model_state_dict"]
    out = {}
    for k, v in sd.items():
        if k in ("initted", "step", "update"):
            continue
        k = k[len("ema_model."):] if k.startswith("ema_model.") else k
        if k.startswith("mel_spec."):
            continue
        out[k] = v
    return out


TEXT_EMB = "transformer.text_embed.text_embed.weight"


def build_model(vocab_map, device="cuda", checkpoint_activations=False):
    import torch  # noqa: F401
    from f5_tts.model.backbones.dit import DiT
    from f5_tts.model.cfm import CFM
    arch, mel = model_cfg()
    arch["checkpoint_activations"] = bool(checkpoint_activations)
    model = CFM(transformer=DiT(**arch, text_num_embeds=len(vocab_map), mel_dim=mel["n_mel_channels"]),
                mel_spec_kwargs=mel, vocab_char_map=vocab_map)
    return model.to(device)


def check_alignment(vocab_map, n_rows, used_chars):
    """官方釋出的 vocab.txt(5559 字元)比權重的文字嵌入表(5555 字元 + 1 列填充)多 4 個字元,不知道是哪 4 個。
    詞表依編碼排序,ASCII 字元全部排在最前面(編號 0 起連續);多出來的字元只要不是 ASCII,就只會讓
    ASCII 之後的編號錯位。所以只要客語拼音用到的字元全是 ASCII,編號就和權重訓練時一致。
    回傳說明文字;對不上又無法確認時報錯。"""
    n_tok = n_rows - 1
    if n_tok == len(vocab_map):
        return "詞表與權重一致"
    toks = sorted(vocab_map, key=vocab_map.get)
    k = 0
    while k < len(toks) and len(toks[k]) == 1 and ord(toks[k]) < 128:
        k += 1
    used = sorted(set(used_chars))
    later_ascii = [t for t in toks[k:] if len(t) == 1 and ord(t) < 128]
    non_ascii = [c for c in used if ord(c) >= 128]
    if n_tok > len(vocab_map) or later_ascii or non_ascii or toks != sorted(toks):
        raise RuntimeError(f"vocab.txt 有 {len(vocab_map)} 個字元,權重的文字嵌入表對應 {n_tok} 個,且無法確認編號一致"
                           f"(用到的非 ASCII 字元 {non_ascii[:10]};排在後面的 ASCII {later_ascii[:10]})")
    return (f"vocab.txt 比權重多 {len(vocab_map) - n_tok} 個字元;詞表依編碼排序、ASCII 在最前面(編號 0–{k - 1}),"
            f"客語拼音只用到 ASCII 字元(最大編號 {max(vocab_map[c] for c in used)}),編號與權重一致")


def load_pretrained(model, ckpt, n_old, log=print, used_chars=None):
    """載入官方權重;文字嵌入表複製權重有的列,其餘的列用舊列的平均與標準差隨機初始化。
    n_old:官方 vocab.txt 的字元數;used_chars:實際會用到的字元(用來確認編號一致)。"""
    import torch
    if str(ckpt).endswith(".safetensors"):
        from safetensors.torch import load_file
        sd = load_file(str(ckpt), device="cpu")
    else:
        sd = torch.load(str(ckpt), map_location="cpu", weights_only=True)
    sd = clean_state(sd)
    old = sd[TEXT_EMB]
    if old.shape[0] != n_old + 1:
        if used_chars is None:
            raise RuntimeError(f"vocab.txt 有 {n_old} 個字元,但 checkpoint 的文字嵌入表有 {old.shape[0]} 列(應為 {n_old + 1})")
        log(check_alignment(model.vocab_char_map, old.shape[0], used_chars))
    cur = model.state_dict()[TEXT_EMB]
    new = cur.clone().float()
    new[: old.shape[0]] = old.float()
    if cur.shape[0] > old.shape[0]:
        g = torch.Generator().manual_seed(0)
        mu, sd_ = old.float().mean(0), old.float().std(0)
        new[old.shape[0]:] = mu + sd_ * torch.randn(cur.shape[0] - old.shape[0], old.shape[1], generator=g)
    sd[TEXT_EMB] = new.to(cur.dtype)
    res = model.load_state_dict(sd, strict=False)
    bad = [k for k in res.missing_keys + res.unexpected_keys if not k.endswith("inv_freq")]   # 旋轉位置編碼是固定值
    if bad:
        raise RuntimeError(f"官方權重與模型不符:missing={res.missing_keys[:5]} unexpected={res.unexpected_keys[:5]}")
    log(f"載入官方權重 {ckpt}(嵌入表 {old.shape[0]} 列沿用,{cur.shape[0] - old.shape[0]} 列新初始化)")
    return model


def load_finetuned(ckpt, device="cuda"):
    """r_train.py 存的 model_<步數>.pt:{ema, vocab, step}。回傳 (model, vocab_map, step)。"""
    import torch
    st = torch.load(str(ckpt), map_location="cpu", weights_only=False)
    vocab = st["vocab"]
    model = build_model(vocab, device)
    res = model.load_state_dict({k: v.float() for k, v in st["ema"].items()}, strict=False)
    sd = model.state_dict()
    bad = [k for k in res.missing_keys if sd[k].dtype.is_floating_point] + list(res.unexpected_keys)
    if bad:
        raise RuntimeError(f"{ckpt} 與模型不符:{bad[:5]}")
    return model.eval(), vocab, st["step"]


def load_vocoder(device="cuda"):
    from f5_tts.model.vocoder import load_hifigan_vocoder
    if not HIFIGAN16K.exists():
        raise FileNotFoundError(f"找不到 {HIFIGAN16K}:先跑 bash r_setup.sh")
    return load_hifigan_vocoder(str(HIFIGAN16K), device=device)


def synthesize(model, vocoder, prompt_audio, prompt_text, text, gen_frames, seed, device="cuda"):
    """F5 式的「接續生成」:聲音提示 + 要說的文字,總長 = 提示的音框數 + gen_frames。回傳 16 kHz 波形(numpy)。"""
    import torch
    a = torch.as_tensor(prompt_audio, dtype=torch.float32)[None].to(device)
    rms = torch.sqrt(torch.mean(a ** 2)).item()
    if rms < TARGET_RMS:
        a = a * TARGET_RMS / max(rms, 1e-8)
    p = mel_frames(a.shape[-1])
    with torch.inference_mode():
        out, _ = model.sample(cond=a, text=[prompt_text + " " + text], duration=p + int(gen_frames),
                              steps=NFE_STEPS, cfg_strength=CFG_STRENGTH, sway_sampling_coef=SWAY, seed=seed)
        mel = out[:, p:, :].float().permute(0, 2, 1)
        wav = vocoder(mel).squeeze().float()
    if rms < TARGET_RMS:
        wav = wav * rms / TARGET_RMS
    return wav.clamp(-1, 1).cpu().numpy()
