"""R1:用客語 1 小時資料微調 Raon-OpenTTS-0.3B(官方英文預訓練權重)。需要 raon 環境與 GPU。

訓練目標與官方相同(F5-TTS 的流匹配填空:隨機遮住 70–100% 的 mel,由文字與其餘 mel 預測被遮住的部分);
沒有時長標籤,也沒有 MAS / MFA 對齊——每個音多長完全由模型內部的注意力自己決定。
超參數事先固定、不依測試集調整:峰值學習率 2e-5、暖身 1,000 步、之後線性降到 0;梯度裁剪 1.0;bf16;
EMA 0.999(存檔與評估都用 EMA 權重)。

    python r_train.py --exp raon_fair1h_sixian_s1234 --dialect sixian --seed 1234
    已有 last.pt 時從中斷處接著訓練。顯示卡記憶體不夠時,該批會自動切半再算(不會略過任何句子)。
"""
import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from hakka_c import raon_io as R  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", required=True)
    ap.add_argument("--dialect", required=True)
    ap.add_argument("--split", default="train")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--max-steps", type=int, default=20000)
    ap.add_argument("--save-steps", default="2500,5000,10000,15000,20000", help="存 EMA 權重的步數(逗號分隔)")
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--warmup", type=int, default=1000)
    ap.add_argument("--max-frames", type=int, default=4000, help="每批 mel 音框上限(含 padding;62.5 音框 = 1 秒)")
    ap.add_argument("--ema", type=float, default=0.999)
    ap.add_argument("--grad-ckpt", action="store_true", help="activation checkpointing(省記憶體、慢約 3 成)")
    ap.add_argument("--log-every", type=int, default=100)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    import soundfile as sf
    import torch

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    out = R.exp_dir(args.exp)
    out.mkdir(parents=True, exist_ok=True)
    log = open(out / "train.log", "a", encoding="utf-8")

    def say(msg):
        print(msg, flush=True)
        log.write(msg + "\n")
        log.flush()

    rows = R.load_jsonl(R.data_dir(args.dialect) / f"{args.split}.jsonl")
    vocab = R.read_vocab(R.RAON / "vocab_hakka.txt")
    n_old = json.load(open(R.RAON / "vocab_hakka.json"))["n_official"]
    audio = [torch.from_numpy(sf.read(r["wav16"], dtype="float32")[0]) for r in rows]
    lens = [R.mel_frames(len(a)) for a in audio]
    if max(lens) > args.max_frames:
        args.max_frames = max(lens)
        say(f"最長的句子有 {max(lens)} 音框,每批上限調成 {args.max_frames}")
    say(f"訓練句數 {len(rows)}({sum(r['sec'] for r in rows) / 3600:.2f} 小時);詞表 {len(vocab)} 字元(官方 {n_old})")

    model = R.build_model(vocab, args.device, checkpoint_activations=args.grad_ckpt)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    ema = None
    step, epoch, pos = 0, 0, 0
    last = out / "last.pt"
    if last.exists():
        st = torch.load(str(last), map_location="cpu", weights_only=False)
        model.load_state_dict(st["model"])
        opt.load_state_dict(st["opt"])
        ema = {k: v.to(args.device) for k, v in st["ema"].items()}
        step, epoch, pos = st["step"], st["epoch"], st["pos"]
        random.setstate(st["py_rng"])
        torch.set_rng_state(st["torch_rng"])
        torch.cuda.set_rng_state(st["cuda_rng"])
        say(f"從第 {step} 步接著訓練")
    else:
        ck, _ = R.find_pretrained()
        R.load_pretrained(model, ck, n_old, log=say)
    if ema is None:
        ema = {k: v.detach().clone().float() for k, v in model.state_dict().items() if v.dtype.is_floating_point}
    save_steps = {int(s) for s in args.save_steps.split(",") if s} | {args.max_steps}
    say(f"參數 {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M;每批 ≤ {args.max_frames} 音框;"
        f"學習率 {args.lr}、暖身 {args.warmup}、共 {args.max_steps} 步;存檔 {sorted(save_steps)}")

    def loss_of(idx):
        a = [audio[i] for i in idx]
        L = max(len(x) for x in a)
        wav = torch.stack([torch.nn.functional.pad(x, (0, L - len(x))) for x in a]).to(args.device)
        with torch.no_grad():
            mel = model.mel_spec(wav).permute(0, 2, 1)
        ml = torch.tensor([lens[i] for i in idx], device=args.device).clamp(max=mel.shape[1])
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss, _, _ = model(mel, [rows[i]["text"] for i in idx], lens=ml)
        return loss.float()

    def run(b):
        """算這批的梯度;記憶體不夠就把最大的一份切半、梯度歸零後整批重算(梯度累加),不略過任何句子。"""
        parts, n_split = [b], 0
        while True:
            opt.zero_grad(set_to_none=True)
            try:
                tot = 0.0
                for idx in parts:
                    w = len(idx) / len(b)
                    loss = loss_of(idx)
                    (loss * w).backward()
                    tot += loss.item() * w
                return tot, n_split
            except torch.cuda.OutOfMemoryError:
                k = max(range(len(parts)), key=lambda j: len(parts[j]))
                if len(parts[k]) == 1:
                    raise
                h = len(parts[k]) // 2
                parts[k:k + 1] = [parts[k][:h], parts[k][h:]]
                n_split += 1
                loss = None
                torch.cuda.empty_cache()

    model.train()
    pnames = {k for k, _ in model.named_parameters()}
    t0, acc, n_acc, n_split = time.time(), 0.0, 0, 0
    while step < args.max_steps:
        batches = R.frame_batches(lens, args.max_frames, args.seed * 1000 + epoch)
        while pos < len(batches) and step < args.max_steps:
            b = batches[pos]
            pos += 1
            step += 1
            for g in opt.param_groups:
                g["lr"] = R.lr_at(step, args.lr, args.warmup, args.max_steps)
            loss, ns = run(b)
            n_split += ns
            if not math.isfinite(loss):
                say(f"★ 第 {step} 步損失為 {loss}(句子 {[rows[i]['stem'] for i in b]})")
                sys.exit(1)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            with torch.no_grad():
                for k, v in model.state_dict().items():
                    if k in ema:
                        ema[k].mul_(args.ema).add_(v.float(), alpha=1 - args.ema)
            acc += loss
            n_acc += 1
            if step % args.log_every == 0 or step == args.max_steps:
                sp = (time.time() - t0) / n_acc
                mem = torch.cuda.max_memory_allocated() / 2 ** 30
                say(f"step {step} loss {acc / n_acc:.4f} lr {opt.param_groups[0]['lr']:.2e} {sp:.2f}s/step "
                    f"剩 {sp * (args.max_steps - step) / 3600:.1f}h 顯存峰值 {mem:.1f}GB 切半 {n_split} 次")
                t0, acc, n_acc = time.time(), 0.0, 0
            if step in save_steps:
                # 參數存 fp16(官方推論也用 fp16),緩衝(例如旋轉位置編碼)存 fp32
                torch.save({"ema": {k: (v.half() if k in pnames else v).cpu() for k, v in ema.items()}, "vocab": vocab,
                            "step": step, "exp": args.exp, "dialect": args.dialect, "seed": args.seed},
                           out / f"model_{step}.pt")
                torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "ema": {k: v.cpu() for k, v in ema.items()},
                            "step": step, "epoch": epoch, "pos": pos, "py_rng": random.getstate(),
                            "torch_rng": torch.get_rng_state(), "cuda_rng": torch.cuda.get_rng_state()}, last)
                say(f"存檔 model_{step}.pt")
        if pos >= len(batches):
            epoch, pos = epoch + 1, 0
    if (out / f"model_{args.max_steps}.pt").exists():
        last.unlink(missing_ok=True)            # 訓練完成就刪掉含優化器狀態的大檔(約 5 GB)
        say("訓練完成")


if __name__ == "__main__":
    main()
