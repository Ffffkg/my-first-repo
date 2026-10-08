"""實驗 C2 第 2 步:由既有的 baseline experiment config 產生「MFA 時長監督」版本(單一變因)。

只改兩件事:data config 加 load_durations: true 並改用 *_mfadur.txt 清單;model 開啟
use_precomputed_durations。其他(步數、驗證間隔、種子、學習率…)一律沿用 baseline。

    python c2_setup.py --base-exp hakka_joint_sixian_ladder_1h
    python c2_setup.py --base-exp fair1h_joint_sixian_scratch_s1234      # 實驗 B 的固定 40k 步規則
"""
import argparse
import re
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from hakka_c import config as C  # noqa: E402

HEADER = "# @package _global_\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-exp", required=True, help="configs/experiment/ 下的 baseline 名稱(不含 .yaml)")
    ap.add_argument("--prefix", default="c2_mfadur_")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    src = (C.MATCHA / "matcha" / "models" / "matcha_tts.py").read_text(encoding="utf-8")
    if "use_precomputed_durations" not in src:
        sys.exit("★ 你的 Matcha 沒有 use_precomputed_durations(舊版)。請把 probe_report.txt 貼回對話,我會提供 patch。")

    exp_dir = C.MATCHA / "configs" / "experiment"
    data_dir = C.MATCHA / "configs" / "data"
    base_path = exp_dir / f"{args.base_exp}.yaml"
    text = base_path.read_text(encoding="utf-8")
    cfg = yaml.safe_load(text)

    data_name = None
    for item in cfg.get("defaults", []):
        if isinstance(item, dict):
            for k, v in item.items():
                if k.replace(" ", "") in ("override/data", "/data"):
                    data_name = str(v).replace(".yaml", "")
    if data_name is None:
        sys.exit(f"★ 在 {base_path} 的 defaults 找不到 'override /data';請把該檔內容貼回對話")

    dcfg = yaml.safe_load((data_dir / f"{data_name}.yaml").read_text(encoding="utf-8"))
    for key in ("train_filelist_path", "valid_filelist_path"):
        p = Path(dcfg[key])
        new = p.with_name(p.stem + "_mfadur.txt")
        full = new if new.is_absolute() else C.MATCHA / new
        if not full.exists():
            sys.exit(f"★ {full} 不存在:先跑 c2_mfa2durations.py(要包含 {p.stem} 這份清單)")
        n_old = sum(1 for _ in open(p if p.is_absolute() else C.MATCHA / p, encoding="utf-8"))
        n_new = sum(1 for _ in open(full, encoding="utf-8"))
        print(f"  {key}: {p.name} → {new.name}({n_new}/{n_old} 句)")
        dcfg[key] = str(new)
    dcfg["load_durations"] = True
    if "name" in dcfg:
        dcfg["name"] = f"{dcfg['name']}_mfadur"
    new_data = f"{data_name}_mfadur"
    new_exp = args.prefix + re.sub(r"^hakka_", "", args.base_exp)

    for path in (data_dir / f"{new_data}.yaml", exp_dir / f"{new_exp}.yaml"):
        if path.exists() and not args.force:
            sys.exit(f"★ {path} 已存在(加 --force 覆寫)")

    (data_dir / f"{new_data}.yaml").write_text(yaml.safe_dump(dcfg, allow_unicode=True, sort_keys=False), encoding="utf-8")

    new_defaults = []
    for item in cfg.get("defaults", []):
        if isinstance(item, dict) and any(k.replace(" ", "") in ("override/data", "/data") for k in item):
            item = {k: new_data for k in item}
        new_defaults.append(item)
    cfg["defaults"] = new_defaults
    cfg.setdefault("model", {})
    cfg["model"]["use_precomputed_durations"] = True
    for key in ("run_name", "experiment", "name"):
        if key in cfg and isinstance(cfg[key], str):
            cfg[key] = new_exp
    if isinstance(cfg.get("tags"), list):
        cfg["tags"] = cfg["tags"] + ["c2_mfadur"]
    (exp_dir / f"{new_exp}.yaml").write_text(HEADER + yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False),
                                             encoding="utf-8")
    print(f"\n已產生:\n  configs/data/{new_data}.yaml\n  configs/experiment/{new_exp}.yaml")
    print("\n先確認設定(Hydra 會印出完整 config,檢查 load_durations 與 use_precomputed_durations 都是 true):")
    print(f"  cd {C.MATCHA} && python matcha/train.py experiment={new_exp} --cfg job | grep -n 'durations\\|filelist'")
    print("\n背景訓練(手冊第 14.1 節:nohup 一定要先 conda activate):")
    print(f"  nohup bash -c 'source ~/miniconda3/etc/profile.d/conda.sh; conda activate tts; "
          f"cd {C.MATCHA}; python matcha/train.py experiment={new_exp}' > {C.HAKKA}/runs/{new_exp}.log 2>&1 &")


if __name__ == "__main__":
    main()
