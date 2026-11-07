"""訓練客語 VITS(單 GPU)。訓練迴圈照官方 train_ms.py(損失、權重、最佳化器、fp16、學習率衰減都相同),
只多了:Matcha 格式的清單與文字前端、--durations mfa(+AD)、--init-from(跨腔初始化)、--dp-kernel / --dp-filter、固定步數。

    # VITS base(MAS)
    python v_train.py --exp vits_base_fair1h_sixian_s1234 --train-list <train.txt> --seed 1234
    # VITS +AD(MFA 時長;清單要用 *_mfadur.txt,時長讀 <音檔>/../../durations/<檔名>.npy)
    python v_train.py --exp vits_ad_fair1h_sixian_s1234 --train-list <train_mfadur.txt> --durations mfa --seed 1234
    # XAD:再加 --init-from <另一腔 48 h 的 G_*.pth> --dp-kernel 5 --dp-filter 512

輸出:~/hakka_tts/vits/<exp>/G_<步數>.pth(只有生成器權重,評估用)、last.pth(續訓用)、config.json、train.log。
同一個 --exp 再跑一次會從 last.pth 接著訓練。
"""
import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from hakka_c import config as C  # noqa: E402

N_SPEAKERS = 4      # joint 語者編號:0/1 四縣、2/3 海陸(與 Matcha 相同)


def read_list(path):
    rows = []
    for line in open(path, encoding="utf-8"):
        line = line.rstrip("\n")
        if line:
            wav, spk, text = line.split("|")[:3]
            w = Path(wav)
            rows.append((w if w.is_absolute() else C.MATCHA / w, int(spk), text))
    return rows


class Data:
    """回傳 (文字 id、語者、波形、MFA 時長或 None)。時長檔與 Matcha 相同:blank 穿插後每個符號的 frame 數。"""

    def __init__(self, rows, durations, dur_dir=None, cleaners="hakka_cleaners"):
        import soundfile as sf
        sys.path.insert(0, str(C.MATCHA))
        from matcha.text import text_to_sequence
        from matcha.utils.utils import intersperse
        self.items, bad = [], 0
        for wav, spk, text in rows:
            seq, _ = text_to_sequence(text, [cleaners])
            x = intersperse(seq, 0)
            n = sf.info(str(wav)).frames
            T = n // C.HOP
            if T <= C.VITS_SEG_FRAMES:                 # 比訓練片段(8192 樣本 = 32 frame)還短的句子官方也無法取片段
                bad += 1
                continue
            d = None
            if durations == "mfa":
                p = (Path(dur_dir) if dur_dir else wav.parent.parent / "durations") / f"{wav.stem}.npy"
                d = np.load(p).astype(np.int64)
                if len(d) != len(x) or abs(int(d.sum()) - T) > 2:
                    bad += 1
                    continue
                d[-1] += T - int(d.sum())          # 頂多差 1–2 frame(音檔長度不是 hop 的整數倍),補在句尾 blank
                if d[-1] < 0:
                    bad += 1
                    continue
            if spk >= N_SPEAKERS:
                raise ValueError(f"語者編號 {spk} 超過 {N_SPEAKERS - 1}")
            self.items.append((np.asarray(x, np.int64), spk, str(wav), T * C.HOP, d))
        if bad:
            print(f"  ★ {bad} 句太短,或時長檔與文字 / 音檔長度對不上,已略過")
        if not self.items:
            sys.exit("★ 沒有可用的訓練句")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        import soundfile as sf
        x, spk, wav, L, d = self.items[i]
        a, sr = sf.read(wav, dtype="float32", always_2d=True)
        assert sr == C.SR, f"{wav} 取樣率 {sr}"
        return x, spk, a.mean(1)[:L], d


def collate(batch):
    import torch
    b = len(batch)
    xl = torch.tensor([len(x) for x, *_ in batch])
    yl = torch.tensor([len(y) for _, _, y, _ in batch])
    X = torch.zeros(b, int(xl.max()), dtype=torch.long)
    Y = torch.zeros(b, int(yl.max()))
    D = torch.zeros(b, int(xl.max()), dtype=torch.long) if batch[0][3] is not None else None
    for i, (x, _, y, d) in enumerate(batch):
        X[i, :len(x)] = torch.from_numpy(x)
        Y[i, :len(y)] = torch.from_numpy(y)
        if D is not None:
            D[i, :len(d)] = torch.from_numpy(d)
    return X, xl, Y, yl, torch.tensor([s for _, s, _, _ in batch]), D


def batches(data, bs, rng):
    """依長度分桶(與官方 DistributedBucketSampler 同目的:減少 padding),每個 epoch 重新洗牌。"""
    idx = sorted(range(len(data)), key=lambda i: data.items[i][3] * (1 + 0.1 * rng.random()))
    out = [idx[k:k + bs] for k in range(0, len(idx), bs)]
    out = [b for b in out if len(b) == bs] or out
    rng.shuffle(out)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", required=True)
    ap.add_argument("--train-list", required=True)
    ap.add_argument("--durations", default="mas", choices=("mas", "mfa"), help="mas = 官方 VITS;mfa = +AD")
    ap.add_argument("--dur-dir", help="預設 <音檔>/../../durations")
    ap.add_argument("--init-from", help="跨腔初始化:另一腔 48 h 模型的 G_*.pth(同資料夾的 D_init.pth 若存在也會載入)")
    ap.add_argument("--dp", default="det", choices=("det", "sdp"), help="det = 確定性 DP(與 Matcha 同構);sdp = 官方預設")
    ap.add_argument("--dp-kernel", type=int, default=3)
    ap.add_argument("--dp-filter", type=int, default=256)
    ap.add_argument("--max-steps", type=int, default=100000)
    ap.add_argument("--save-every", type=int, default=10000)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--fp16", type=int, default=1, help="官方設定為 fp16 混合精度")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--log-every", type=int, default=200)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    import torch
    from torch.nn import functional as F
    from hakka_c import vits_model as V
    _, commons, losses = V.import_vits()
    H = V.TRAIN_HPS

    out = C.VITS_RUNS / args.exp
    out.mkdir(parents=True, exist_ok=True)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    dev = torch.device(args.device)
    torch.backends.cudnn.benchmark = True

    data = Data(read_list(args.train_list), args.durations, args.dur_dir)
    print(f"訓練句數 {len(data)}({args.train_list});時長來源 {args.durations}")
    net_g, net_d = V.build(V.n_vocab(), N_SPEAKERS, args.dp, args.dp_kernel, args.dp_filter)
    net_g, net_d = net_g.to(dev), net_d.to(dev)
    cfg = {"args": vars(args), "model": V.MODEL_HPS, "train": H, "n_vocab": V.n_vocab(), "n_speakers": N_SPEAKERS,
           "vits_commit": V.VITS_COMMIT}
    opt_g = torch.optim.AdamW(net_g.parameters(), H["learning_rate"], betas=H["betas"], eps=H["eps"])
    opt_d = torch.optim.AdamW(net_d.parameters(), H["learning_rate"], betas=H["betas"], eps=H["eps"])
    scaler = torch.amp.GradScaler("cuda", enabled=bool(args.fp16))

    step, epoch = 0, 0
    last = out / "last.pth"
    if last.exists():
        st = torch.load(last, map_location=dev, weights_only=False)
        net_g.load_state_dict(st["G"]); net_d.load_state_dict(st["D"])
        opt_g.load_state_dict(st["opt_g"]); opt_d.load_state_dict(st["opt_d"])
        scaler.load_state_dict(st["scaler"])
        step, epoch = st["step"], st["epoch"]
        print(f"從 {last} 接著訓練(第 {step} 步)")
    elif args.init_from:
        src = Path(args.init_from).expanduser()
        V.load_transfer(net_g, torch.load(src, map_location=dev, weights_only=False)["G"], "生成器")
        d_src = src.parent / "D_last.pth"
        if d_src.exists():
            V.load_transfer(net_d, torch.load(d_src, map_location=dev, weights_only=False)["D"], "判別器")
        cfg["transfer_from"] = str(src)
    json.dump(cfg, open(out / "config.json", "w"), indent=2, ensure_ascii=False)

    loader_rng = random.Random(args.seed)
    for _ in range(epoch):            # 續訓時重現洗牌順序
        batches(data, args.batch_size, loader_rng)
    log = open(out / "train.log", "a", encoding="utf-8")
    seg = H["segment_size"] // C.HOP
    t0, t_last = time.time(), time.time()
    net_g.train(); net_d.train()
    while step < args.max_steps:
        dl = torch.utils.data.DataLoader(data, batch_sampler=batches(data, args.batch_size, loader_rng),
                                         collate_fn=collate, num_workers=args.workers, pin_memory=dev.type == "cuda")
        lr = H["learning_rate"] * H["lr_decay"] ** epoch      # 官方:每個 epoch 衰減一次
        for o in (opt_g, opt_d):
            for g in o.param_groups:
                g["lr"] = lr
        for X, xl, Y, yl, sid, D in dl:
            X, xl, Y, yl, sid = (t.to(dev, non_blocking=True) for t in (X, xl, Y, yl, sid))
            D = D.to(dev) if D is not None else None
            with torch.no_grad():
                spec = V.spectrogram(Y)
                sl = yl // C.HOP
                mel = V.spec_to_mel(spec)
            with torch.autocast("cuda", dtype=torch.float16, enabled=bool(args.fp16)):
                y_hat, l_length, attn, ids, x_mask, z_mask, (z, z_p, m_p, logs_p, m_q, logs_q) = \
                    net_g(X, xl, spec, sl, sid, dur=D)
                y_mel = commons.slice_segments(mel, ids, seg)
                y_hat_mel = V.mel_from_wav(y_hat.squeeze(1).float())
                y = commons.slice_segments(Y.unsqueeze(1), ids * C.HOP, H["segment_size"])
                y_d_r, y_d_g, _, _ = net_d(y, y_hat.detach())
                with torch.autocast("cuda", enabled=False):
                    loss_disc, _, _ = losses.discriminator_loss(y_d_r, y_d_g)
            opt_d.zero_grad()
            scaler.scale(loss_disc).backward()
            scaler.unscale_(opt_d)
            scaler.step(opt_d)
            with torch.autocast("cuda", dtype=torch.float16, enabled=bool(args.fp16)):
                y_d_r, y_d_g, fmap_r, fmap_g = net_d(y, y_hat)
                with torch.autocast("cuda", enabled=False):
                    loss_dur = torch.sum(l_length.float())
                    loss_mel = F.l1_loss(y_mel, y_hat_mel) * H["c_mel"]
                    loss_kl = losses.kl_loss(z_p, logs_q, m_p, logs_p, z_mask) * H["c_kl"]
                    loss_fm = losses.feature_loss(fmap_r, fmap_g)
                    loss_gen, _ = losses.generator_loss(y_d_g)
                    loss_g = loss_gen + loss_fm + loss_mel + loss_dur + loss_kl
            opt_g.zero_grad()
            scaler.scale(loss_g).backward()
            scaler.unscale_(opt_g)
            scaler.step(opt_g)
            scaler.update()
            step += 1
            if step % args.log_every == 0:
                dt = (time.time() - t_last) / args.log_every
                t_last = time.time()
                msg = (f"step {step} epoch {epoch} lr {lr:.2e} | d {loss_disc.item():.3f} g {loss_gen.item():.3f} "
                       f"fm {loss_fm.item():.3f} mel {loss_mel.item():.3f} dur {loss_dur.item():.4f} "
                       f"kl {loss_kl.item():.3f} | {dt:.3f} s/step,剩 {(args.max_steps - step) * dt / 3600:.1f} h")
                print(msg, flush=True)
                log.write(msg + "\n"); log.flush()
                if not all(map(np.isfinite, (loss_g.item(), loss_disc.item()))):
                    sys.exit(f"★ 第 {step} 步損失不是有限值")
            if step % args.save_every == 0 or step == args.max_steps:
                torch.save({"G": net_g.state_dict(), "step": step, "config": cfg}, out / f"G_{step}.pth")
                torch.save({"G": net_g.state_dict(), "D": net_d.state_dict(), "opt_g": opt_g.state_dict(),
                            "opt_d": opt_d.state_dict(), "scaler": scaler.state_dict(), "step": step,
                            "epoch": epoch + 1}, last)
                torch.save({"D": net_d.state_dict()}, out / "D_last.pth")
                print(f"  存檔 G_{step}.pth({(time.time() - t0) / 3600:.1f} h)", flush=True)
            if step >= args.max_steps:
                break
        epoch += 1
    print(f"完成:{out}")


if __name__ == "__main__":
    main()
