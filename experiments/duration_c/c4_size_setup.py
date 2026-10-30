"""C4 資料量實驗:由 1h 的 baseline 設定(fair1h_*)產生其他資料量的 baseline 設定(單一變因:訓練清單)。

只換 train_filelist_path(例如 ladder_1h → ladder_10h),其他全部沿用 1h 設定:
驗證集、固定 40,000 步、每 1,000 步驗證、固定步數存檔、種子、學習率,以及 mel 統計量
(baseline 與 C2 用同一組統計量,比較公平;評估時 c_eval 讀模型內的統計量,與訓練一致)。
產生後再用 c2_setup.py 從新的 baseline 產生 C2 版本。

    python c4_size_setup.py --base-exp fair1h_joint_sixian_scratch_s1234 --train-list ladder_10h --tag fair10h
    python c2_setup.py --base-exp fair10h_joint_sixian_scratch_s1234
"""
import argparse
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from hakka_c import config as C  # noqa: E402

HEADER = "# @package _global_\n"


def data_name_of(cfg):
    for item in cfg.get("defaults", []):
        if isinstance(item, dict):
            for k, v in item.items():
                if k.replace(" ", "") in ("override/data", "/data"):
                    return str(v).replace(".yaml", "")
    return None


def replace_strings(obj, old, new):
    if isinstance(obj, str):
        return obj.replace(old, new)
    if isinstance(obj, list):
        return [replace_strings(x, old, new) for x in obj]
    if isinstance(obj, dict):
        return {k: replace_strings(v, old, new) for k, v in obj.items()}
    return obj


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-exp", required=True, help="1h baseline,例如 fair1h_joint_sixian_scratch_s1234")
    ap.add_argument("--train-list", required=True, help="新的訓練清單名稱(不含 .txt),例如 ladder_10h")
    ap.add_argument("--tag", required=True, help="取代名稱裡的 fair1h,例如 fair10h")
    ap.add_argument("--old-tag", default="fair1h")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    exp_dir, data_dir = C.MATCHA / "configs" / "experiment", C.MATCHA / "configs" / "data"
    base_path = exp_dir / f"{args.base_exp}.yaml"
    if args.old_tag not in args.base_exp:
        sys.exit(f"★ {args.base_exp} 名稱裡沒有 {args.old_tag}")
    cfg = yaml.safe_load(base_path.read_text(encoding="utf-8"))
    data_name = data_name_of(cfg)
    if data_name is None:
        sys.exit(f"★ {base_path} 的 defaults 找不到 'override /data'")
    dcfg = yaml.safe_load((data_dir / f"{data_name}.yaml").read_text(encoding="utf-8"))

    old_train = Path(dcfg["train_filelist_path"])
    new_train = old_train.with_name(f"{args.train_list}.txt")
    full = new_train if new_train.is_absolute() else C.MATCHA / new_train
    if not full.exists():
        sys.exit(f"★ {full} 不存在;同資料夾有:{sorted(p.name for p in full.parent.glob('*.txt'))}")
    n_old = sum(1 for _ in open(old_train if old_train.is_absolute() else C.MATCHA / old_train, encoding="utf-8"))
    n_new = sum(1 for _ in open(full, encoding="utf-8"))
    print(f"  訓練清單:{old_train.name}({n_old} 句)→ {new_train.name}({n_new} 句);驗證集不變:{Path(dcfg['valid_filelist_path']).name}")

    new_data = data_name.replace(args.old_tag, args.tag) if args.old_tag in data_name else f"{data_name}_{args.tag}"
    new_exp = args.base_exp.replace(args.old_tag, args.tag)
    dcfg["train_filelist_path"] = str(new_train)
    if "name" in dcfg:
        dcfg["name"] = new_data
    cfg = replace_strings(cfg, args.base_exp, new_exp)
    cfg["defaults"] = [({k: new_data for k in item} if isinstance(item, dict) and data_name_of({"defaults": [item]})
                        else item) for item in cfg.get("defaults", [])]
    if isinstance(cfg.get("tags"), list):
        cfg["tags"] = [t.replace(args.old_tag, args.tag) if isinstance(t, str) else t for t in cfg["tags"]]

    for path in (data_dir / f"{new_data}.yaml", exp_dir / f"{new_exp}.yaml"):
        if path.exists() and not args.force:
            sys.exit(f"★ {path} 已存在(加 --force 覆寫)")
    (data_dir / f"{new_data}.yaml").write_text(yaml.safe_dump(dcfg, allow_unicode=True, sort_keys=False), encoding="utf-8")
    (exp_dir / f"{new_exp}.yaml").write_text(HEADER + yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False), encoding="utf-8")
    print(f"已產生:configs/data/{new_data}.yaml、configs/experiment/{new_exp}.yaml")
    print(f"檢查:cd {C.MATCHA} && python matcha/train.py experiment={new_exp} --cfg job | grep -n 'filelist\\|max_steps\\|seed\\|data_statistics' -A2")


if __name__ == "__main__":
    main()
