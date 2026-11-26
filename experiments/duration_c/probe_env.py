"""第 0 步:檢查本機環境與資料,輸出 probe_report.txt。

我(Claude)看不到你的電腦,這支程式把我寫程式時「假設」的每一件事實際檢查一遍。
跑完把 probe_report.txt 的內容貼回對話,有不符的地方我再調整程式。

    conda activate tts
    cd <repo>/experiments/duration_c
    python probe_env.py
"""
import inspect
import io
import platform
import re
import sys
import traceback
from collections import Counter
from contextlib import redirect_stdout
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from hakka_c import config as C  # noqa: E402

buf = io.StringIO()


def section(title):
    print(f"\n==== {title} ====")


def safe(fn):
    try:
        fn()
    except Exception:  # noqa: BLE001
        print("  [錯誤]", traceback.format_exc().strip().splitlines()[-1])


def env():
    print("python", platform.python_version(), sys.executable)
    try:
        import torch
        print("torch", torch.__version__, "cuda", torch.cuda.is_available(),
              torch.cuda.get_device_name(0) if torch.cuda.is_available() else "")
    except Exception as e:  # noqa: BLE001
        print("torch 無法 import:", e)
    for m in ("numpy", "scipy", "soundfile", "yaml", "pyworld", "pysptk", "librosa", "parselmouth", "pandas"):
        try:
            mod = __import__(m)
            print(f"  {m:12s} {getattr(mod, '__version__', 'ok')}")
        except Exception:  # noqa: BLE001
            print(f"  {m:12s} (沒有安裝)")


def paths():
    for name in ("HAKKA", "MATCHA", "ANALYSIS", "BENCH", "VOCAB_JSON"):
        p = getattr(C, name)
        print(f"  {name:10s} {p}  {'存在' if p.exists() else '★不存在'}")
    for d in C.DIALECTS:
        tg = C.textgrid_dir(d)
        print(f"  textgrids/{d}: {'存在' if tg.exists() else '★不存在'}", end="")
        if tg.exists():
            print(f",TextGrid 數 {sum(1 for _ in tg.rglob('*.TextGrid'))}")
        else:
            print()
        w = C.HAKKA / "data" / d / "wav22k"
        print(f"  data/{d}/wav22k: {'存在' if w.exists() else '★不存在'}"
              + (f",子資料夾 {[p.name for p in w.iterdir() if p.is_dir()][:8]}" if w.exists() else ""))


def matcha():
    v = C.MATCHA / "matcha" / "VERSION"
    print("  Matcha VERSION:", v.read_text().strip() if v.exists() else "(沒有 VERSION 檔)")
    src = (C.MATCHA / "matcha" / "models" / "matcha_tts.py").read_text(encoding="utf-8")
    print("  MatchaTTS 支援 use_precomputed_durations:", "use_precomputed_durations" in src)
    print("  forward 有 durations 參數:", bool(re.search(r"def forward\([^)]*durations", src)))
    print("  forward 回傳 attn:", "return dur_loss, prior_loss, diff_loss, attn" in src)
    print("  synthesise 用 ceil:", "torch.ceil(w)" in src)
    mcfg = C.MATCHA / "configs" / "model" / "matcha.yaml"
    if mcfg.exists():
        print("  configs/model/matcha.yaml 有 use_precomputed_durations:",
              "use_precomputed_durations" in mcfg.read_text())
    dm = (C.MATCHA / "matcha" / "data" / "text_mel_datamodule.py").read_text(encoding="utf-8")
    print("  datamodule 有 get_durations:", "def get_durations" in dm,
          "| 讀音檔用:", "torchaudio" if "ta.load" in dm else ("soundfile" if "sf.read" in dm else "?"))
    print("  get_durations_from_trained_model.py 存在:",
          (C.MATCHA / "matcha" / "utils" / "get_durations_from_trained_model.py").exists())
    te = (C.MATCHA / "matcha" / "models" / "components" / "text_encoder.py").read_text(encoding="utf-8")
    print("  TextEncoder 有 tone_aware:", "tone_aware" in te, "| proj_w 前 detach:", "torch.detach(x)" in te)
    dc = sorted((C.MATCHA / "configs" / "data").glob("*hakka*.yaml")) + sorted((C.MATCHA / "configs" / "data").glob("fair1h*.yaml"))
    print("  hakka data configs:", [p.name for p in dc][:30])
    if dc:
        print(f"  --- {dc[0].name} ---")
        print("    " + dc[0].read_text(encoding="utf-8").replace("\n", "\n    "))
    ec = sorted((C.MATCHA / "configs" / "experiment").glob("hakka_joint_*ladder*.yaml"))
    print("  experiment configs (joint ladder):", [p.name for p in ec][:20])
    if ec:
        print(f"  --- {ec[0].name} ---")
        print("    " + ec[0].read_text(encoding="utf-8").replace("\n", "\n    "))


def vocab():
    from hakka_c.pua import resolve_pua_map
    m, src = resolve_pua_map()
    print(f"  採用:{src}")
    print("  範例:", [(f"U+{ord(a):04X}", b) for a, b in sorted(m.items())[:15]])
    sys.path.insert(0, str(C.MATCHA))
    from matcha.text import cleaners
    from matcha.text.symbols import symbols as syms
    print("  matcha.text.symbols 長度:", len(syms))
    print("  cleaners 有 hakka_cleaners:", hasattr(cleaners, "hakka_cleaners"))
    cands = sorted(C.BENCH.glob("*.json")) + sorted(C.HAKKA.glob("*/*vocab*.json"))
    print("  benchmark 等處的 JSON:", [str(p.relative_to(C.HAKKA)) for p in cands][:15])


def structure():
    from hakka_c.pua import resolve_pua_map
    from hakka_c.structure import StructError, build_utt
    from hakka_c.textgrid import read_textgrid
    pua, _ = resolve_pua_map(verbose=False)
    sys.path.insert(0, str(C.MATCHA))
    from matcha.text import text_to_sequence
    for d in C.DIALECTS:
        fl = C.filelist("joint", d, "test")
        if not fl.exists():
            print(f"  ★ {fl} 不存在")
            continue
        print(f"  {d} filelists:", sorted(p.name for p in fl.parent.glob("*.txt")))
        lines = [l.rstrip("\n").split("|") for l in open(fl, encoding="utf-8") if l.strip()]
        print(f"  {d} test.txt:{len(lines)} 行;第一行路徑 {lines[0][0]} 語者 {lines[0][1]}")
        tg = {p.stem: p for p in C.textgrid_dir(d).rglob("*.TextGrid")}
        ok, why, shown = 0, Counter(), False
        for wav, spk, text in lines[:40]:
            stem = Path(wav).stem
            try:
                _, clean = text_to_sequence(text, ["hakka_cleaners"])
                names = [pua[c] for c in clean]
                if stem not in tg:
                    raise StructError("找不到 TextGrid")
                tiers = read_textgrid(tg[stem])
                u = build_utt(stem, names, tiers)
                ok += 1
                if not shown:
                    shown = True
                    print("    符號:", names[:20])
                    print("    tiers:", list(tiers))
                    for s in u.syllables[:6]:
                        print(f"    音節 {s.idx} {s.word:8s} 入聲={s.entering} 韻尾={s.coda!r} 母音 {s.ref_ms:.1f} ms 停頓前={s.prepausal}")
            except Exception as e:  # noqa: BLE001
                why[str(e)[:80]] += 1
        print(f"    前 40 句結構對應成功 {ok}/40;失敗原因:{dict(why.most_common(5))}")


def audio_frames():
    import soundfile as sf
    d = C.DIALECTS[0]
    wav = open(C.filelist("joint", d, "test"), encoding="utf-8").readline().split("|")[0]
    a, sr = sf.read(wav)
    print(f"  {Path(wav).name}: sr={sr}, 樣本數 N={len(a)}, N//256={len(a)//256}")
    try:
        import torch
        sys.path.insert(0, str(C.MATCHA))
        from matcha.utils.audio import mel_spectrogram
        m = mel_spectrogram(torch.tensor(a, dtype=torch.float32)[None], C.N_FFT, C.N_MELS, C.SR, C.HOP, C.WIN,
                            C.F_MIN, C.F_MAX, center=False)
        print("  Matcha mel 長度:", m.shape[-1], "(應等於 N//256)")
    except Exception as e:  # noqa: BLE001
        print("  無法計算 mel:", e)


def ckpts():
    from hakka_c.matcha_io import find_ckpt
    for d in C.DIALECTS:
        for h in (1, 5, 10, 30, 48):
            exp = f"hakka_joint_{d}_ladder_{h}h"
            try:
                print(f"  {exp}: {find_ckpt(exp)}")
            except Exception as e:  # noqa: BLE001
                print(f"  {exp}: ★ {e}")
    names = sorted(p.name for p in (C.MATCHA / "logs" / "train").glob("*"))
    print("  logs/train 下的實驗:", names[:60])


def analysis_scripts():
    sys.path.insert(0, str(C.ANALYSIS))
    sys.path.insert(0, str(C.MATCHA))
    try:
        import evaluate_v2 as ev
        for fn in ("analyze_wav", "mcd_from_mcep", "f0_metrics"):
            f = getattr(ev, fn, None)
            print(f"  evaluate_v2.{fn}:", inspect.signature(f) if f else "★沒有")
            if f:
                src = inspect.getsource(f).splitlines()
                print("      " + "\n      ".join(src[:3] + [l for l in src if "return" in l][:2]))
    except Exception as e:  # noqa: BLE001
        print("  evaluate_v2 無法 import:", e)
    p = C.ANALYSIS / "resample_audio.py"
    if p.exists():
        lines = p.read_text(encoding="utf-8").splitlines()
        hits = [f"{i+1}: {l.strip()}" for i, l in enumerate(lines)
                if re.search(r"margin|offset|trim|start|csv|json|log", l, re.I)]
        print("  resample_audio.py 相關行:")
        print("    " + "\n    ".join(hits[:40]))
    else:
        print("  ★ resample_audio.py 不在", C.ANALYSIS)
    for name in ("diag_duration.py", "hakka_frontend.py", "eval_multigen.py"):
        print(f"  {name}:", "存在" if (C.ANALYSIS / name).exists() else "★不存在")


def selfcheck():
    from hakka_c.matcha_io import Engine, find_ckpt, read_filelist
    d = C.DIALECTS[0]
    ck = find_ckpt(f"hakka_joint_{d}_ladder_48h")
    eng = Engine(ck)
    r = read_filelist(C.filelist("joint", d, "test"))[0]
    ok, msg = eng.selfcheck(r["text"], r["spk"])
    print(f"  自寫合成路徑 vs model.synthesise():{'一致 ✓' if ok else '★不一致'}({msg});mel 統計量來源 {eng.stats_src}")
    import yaml
    dc = yaml.safe_load(open(C.MATCHA / "configs" / "data" / f"hakka_joint_{d}_ladder_48h.yaml", encoding="utf-8"))
    print(f"  mel buffer ({eng.mel_mean:.4f}, {eng.mel_std:.4f}) vs data config "
          f"({dc['data_statistics']['mel_mean']}, {dc['data_statistics']['mel_std']})")


if __name__ == "__main__":
    with redirect_stdout(buf):
        for title, fn in [("環境", env), ("路徑", paths), ("Matcha", matcha), ("PUA 對照表", vocab),
                          ("符號 ↔ TextGrid 對應", structure), ("音檔與 frame 數", audio_frames),
                          ("checkpoint", ckpts), ("既有分析腳本", analysis_scripts), ("合成路徑自我檢查", selfcheck)]:
            section(title)
            safe(fn)
    out = buf.getvalue()
    print(out)
    Path("probe_report.txt").write_text(out, encoding="utf-8")
    print("\n→ 已寫入 probe_report.txt,請把內容貼回對話。")
