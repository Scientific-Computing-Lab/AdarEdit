import argparse
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


LINE_RE = re.compile(
    r"^\[(?P<variant>[^\]]+)\]\s+"
    r"(?P<train>[^\s]+)_train->(?P<val>[^\s]+)_valid\s+"
    r"ep\s+(?P<epoch>\d+)\s+f1=(?P<f1>[0-9.]+)"
)


@dataclass(frozen=True)
class Point:
    epoch: int
    f1: float


def palette(n: int) -> List[Tuple[float, float, float]]:
    # Mirror ADARsummary.py's palette for consistent color choice.
    phi = (1 + 5**0.5) / 2
    hues = [(i / phi) % 1.0 for i in range(n)]
    return [tuple(plt.colormaps.get_cmap("hsv")(h)[:3]) for h in hues]


def parse_log(path: Path) -> Dict[Tuple[str, str, str], List[Point]]:
    series: Dict[Tuple[str, str, str], List[Point]] = {}
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            m = LINE_RE.match(line.strip())
            if not m:
                continue
            variant = m.group("variant")
            train = m.group("train")
            val = m.group("val")
            epoch = int(m.group("epoch"))
            f1 = float(m.group("f1"))
            key = (variant, train, val)
            series.setdefault(key, []).append(Point(epoch=epoch, f1=f1))
    return series


def sort_keys(
    keys: Iterable[Tuple[str, str, str]], tissue_order: List[str]
) -> List[Tuple[str, str, str]]:
    rank = {t: i for i, t in enumerate(tissue_order)}

    def key_fn(k: Tuple[str, str, str]) -> Tuple[int, str, int, str]:
        _, train, val = k
        return (rank.get(train, 10**6), train, rank.get(val, 10**6), val)

    return sorted(keys, key=key_fn)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", required=True, help="Log file with per-epoch F1 lines.")
    ap.add_argument("--out_png", required=True, help="Path to write the PNG figure.")
    ap.add_argument(
        "--out_csv",
        default="",
        help="Optional CSV (long format): variant,train,val,epoch,f1.",
    )
    ap.add_argument(
        "--variant",
        default="bioaware_plain",
        help="Filter to a variant name (empty = all).",
    )
    ap.add_argument(
        "--tissue_order",
        default="Liver,Brain_Cerebellum,Muscle_Skeletal,Artery_Tibial,Combined",
        help="Preferred tissue ordering for legend/plot.",
    )
    ap.add_argument("--alpha", type=float, default=0.85)
    ap.add_argument("--lw", type=float, default=1.4)
    ap.add_argument(
        "--smooth_window",
        type=int,
        default=0,
        help="Optional moving-average window (odd integer). 0 disables smoothing.",
    )
    args = ap.parse_args()

    log_path = Path(args.log)
    out_png = Path(args.out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)

    series = parse_log(log_path)
    if args.variant:
        series = {k: v for k, v in series.items() if k[0] == args.variant}

    if not series:
        raise SystemExit("No per-epoch lines matched. Check --log/--variant.")

    tissue_order = [t.strip() for t in args.tissue_order.split(",") if t.strip()]
    keys = sort_keys(series.keys(), tissue_order)
    colors = palette(len(keys))

    if args.smooth_window < 0:
        raise SystemExit("--smooth_window must be >= 0")
    if args.smooth_window and args.smooth_window % 2 == 0:
        raise SystemExit("--smooth_window must be odd (e.g., 5, 9, 15)")

    plt.figure(figsize=(12, 7))
    for idx, key in enumerate(keys):
        variant, train, val = key
        pts = sorted(series[key], key=lambda p: p.epoch)
        epochs = [p.epoch for p in pts]
        f1s = [p.f1 for p in pts]
        if args.smooth_window and len(f1s) >= args.smooth_window:
            half = args.smooth_window // 2
            smoothed = []
            for i in range(len(f1s)):
                lo = max(0, i - half)
                hi = min(len(f1s), i + half + 1)
                smoothed.append(sum(f1s[lo:hi]) / (hi - lo))
            f1s = smoothed
        label = f"{train.replace('_', ' ')} -> {val.replace('_', ' ')}"
        plt.plot(epochs, f1s, color=colors[idx], alpha=args.alpha, lw=args.lw, label=label)

    plt.xlabel("Epoch")
    plt.ylabel("F1")
    title_variant = args.variant if args.variant else "all variants"
    if args.smooth_window:
        plt.title(f"F1 over epochs (smoothed, w={args.smooth_window}) ({title_variant})")
    else:
        plt.title(f"F1 over epochs ({title_variant})")
    plt.grid(True, alpha=0.2)
    plt.legend(fontsize=8, ncol=2)
    plt.tight_layout()
    plt.savefig(out_png, dpi=300)

    if args.out_csv:
        out_csv = Path(args.out_csv)
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        with out_csv.open("w", encoding="utf-8") as f:
            f.write("variant,train,val,epoch,f1\n")
            for key in keys:
                variant, train, val = key
                for p in sorted(series[key], key=lambda p: p.epoch):
                    f.write(f"{variant},{train},{val},{p.epoch},{p.f1}\n")


if __name__ == "__main__":
    main()
