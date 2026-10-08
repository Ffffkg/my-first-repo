"""時長的換算:blank 分配規則、Matcha 的取整方式、MFA → 每符號 frame 數。只用 numpy。"""
import numpy as np

from .config import HOP, SR

RULES = ("half", "left", "right", "none")


def merge_blank(d_all, rule="half"):
    """intersperse 後的時長 [b, p0, b, p1, ..., p_{n-1}, b](長度 2n+1)→ 每個真符號的時長(長度 n)。

    half : 手冊 merge_blank_durations() 的規則。內部 blank 左右各半,頭尾 blank 全給邊界音素。
    left : 內部 blank 全給左邊音素(句首 blank 給第一個音素)。
    right: 內部 blank 全給右邊音素(句尾 blank 給最後一個音素)。
    none : 不分配 blank,只看符號本身(下界)。
    """
    d_all = np.asarray(d_all, dtype=float)
    L = len(d_all)
    assert L % 2 == 1, f"長度應為 2n+1,得到 {L}"
    n = (L - 1) // 2
    out = d_all[1::2].copy()
    if rule == "none" or n == 0:
        return out
    blanks = d_all[0::2]                     # n+1 個:blanks[i] 在符號 i 的左邊
    if rule == "half":
        left_share = np.r_[1.0, np.full(n - 1, 0.5)]     # blank i 給符號 i 的比例
        right_share = np.r_[np.full(n - 1, 0.5), 1.0]    # blank i+1 給符號 i 的比例
        out += blanks[:-1] * left_share + blanks[1:] * right_share
    elif rule == "left":
        out += blanks[1:]                    # 每個符號拿走右邊的 blank
        out[0] += blanks[0]
    elif rule == "right":
        out += blanks[:-1]                   # 每個符號拿走左邊的 blank
        out[-1] += blanks[-1]
    else:
        raise ValueError(rule)
    return out


def matcha_frames(w, length_scale=1.0, mode="ceil"):
    """把時長預測器輸出的連續時長 w(frame)換成每個符號實際分到的整數 frame。

    ceil: 完全重現 Matcha synthesise():w_ceil = ceil(w) * length_scale,
          generate_path 以累積和做 sequence_mask,總長 = floor(sum(w_ceil))。
    cum : 累積四捨五入(總和守恆、不系統性偏長),C1 用來檢驗 ceil 造成的偏差。
    """
    w = np.asarray(w, dtype=float)
    if mode == "ceil":
        # 與 torch 一樣用 float32;arange(T) < cum 的個數恰為 ceil(cum)
        d = np.ceil(w.astype(np.float32)) * np.float32(length_scale)
        total = max(1, int(np.floor(d.sum(dtype=np.float32))))
        c = np.minimum(np.ceil(np.cumsum(d, dtype=np.float32)), total)
    elif mode == "cum":
        c = np.round(np.cumsum(w * length_scale))
    else:
        raise ValueError(mode)
    return np.diff(np.r_[0.0, c]).astype(int)


def sec_to_frame(t, T, sr=SR, hop=HOP):
    """Matcha 的 mel(center=False、左右各補 (n_fft-hop)/2)第 i 格對應 [i·hop, (i+1)·hop) 個樣本。"""
    return int(np.clip(round(t * sr / hop), 0, T))


def mfa_to_frames(n_sym, phone_sym, phone_iv, T, pause_pos=(), offset_s=0.0):
    """MFA 音素邊界 → intersperse 後每個符號(含 blank)的 frame 數,給 Matcha 的 load_durations 用。

    n_sym     : 非 blank 符號數 n(結果長度 2n+1)
    phone_sym : 第 k 個 MFA 音素對應的符號位置
    phone_iv  : 第 k 個 MFA 音素 (start_s, end_s),原始 TextGrid 座標
    T         : 該句梅爾頻譜長度(= 22 kHz 音檔樣本數 // 256)
    pause_pos : 停頓符號的位置
    offset_s  : 22 kHz 音檔相對 TextGrid 的裁切起點(秒);TextGrid 時間要減掉它

    規則:
      * 音素 = 它的 MFA 區間。
      * 兩個音素之間的靜音:中間有停頓符號就給停頓符號(多個則平分),否則給中間那個 blank。
      * 句首/句尾靜音同樣處理。
      * Matcha 的時長損失取 log(frame 數),0 frame 會變成 log(1e-8),所以每個符號至少 1 frame:
        不足的 blank 向「左邊」音素借 1 frame(左邊不夠才向右、再往外找)。
        如此一來,內部音素在 merge_blank(rule='half') 之後恰好還原成 MFA 的 frame 數。
    """
    L = 2 * n_sym + 1
    fr = np.zeros(L, dtype=np.int64)
    pause_pos = set(pause_pos)
    st = [sec_to_frame(a - offset_s, T) for a, _ in phone_iv]
    en = [sec_to_frame(b - offset_s, T) for _, b in phone_iv]
    for k in range(1, len(st)):              # 取整後保持單調
        st[k] = max(st[k], en[k - 1])
    for k in range(len(st)):
        en[k] = max(en[k], st[k])
    for k, i in enumerate(phone_sym):
        fr[2 * i + 1] = en[k] - st[k]

    def put_gap(g, lo, hi):
        """把 g 個 frame 放到「符號 lo 與符號 hi 之間」(lo=-1 表句首、hi=n 表句尾)。"""
        if g <= 0:
            return
        pos = list(range(2 * (lo + 1), 2 * hi + 1))
        pauses = [p for p in pos if p % 2 == 1 and (p - 1) // 2 in pause_pos]
        if pauses:
            q, r = divmod(g, len(pauses))
            for j, p in enumerate(pauses):
                fr[p] += q + (1 if j < r else 0)
        else:
            blanks = [p for p in pos if p % 2 == 0]
            fr[blanks[len(blanks) // 2]] += g

    if phone_sym:
        put_gap(st[0], -1, phone_sym[0])
        for k in range(len(phone_sym) - 1):
            put_gap(st[k + 1] - en[k], phone_sym[k], phone_sym[k + 1])
        put_gap(T - en[-1], phone_sym[-1], n_sym)
    else:
        fr[L // 2] = T

    if T < L:
        raise ValueError(f"梅爾長度 {T} 小於符號數 {L},無法讓每個符號至少 1 frame")
    for p in range(L):
        if fr[p] >= 1:
            continue
        donor = None
        for q in (p - 1, p + 1):
            if 0 <= q < L and fr[q] >= 2:
                donor = q
                break
        if donor is None:
            for dist in range(2, L):
                for q in (p - dist, p + dist):
                    if 0 <= q < L and fr[q] >= 2:
                        donor = q
                        break
                if donor is not None:
                    break
        fr[donor] -= 1
        fr[p] += 1
    assert fr.sum() == T and fr.min() >= 1, (fr.sum(), T, fr.min())
    return fr
