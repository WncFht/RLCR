import argparse
import json
from pathlib import Path

import pandas as pd

CANON_COLS = ["dataset", "model", "acc", "auroc", "brier", "ece"]


def _canonical_metrics(metrics: dict) -> dict:
    acc = metrics.get("acc")
    if acc is None:
        acc = metrics.get("accuracy")
    if acc is None:
        acc = metrics.get("pass@1")

    auroc = metrics.get("auroc")

    brier = metrics.get("brier")
    if brier is None:
        brier = metrics.get("brier_score")

    ece = metrics.get("ece")

    return {"acc": acc, "auroc": auroc, "brier": brier, "ece": ece}


def _read_metrics_file(path: Path) -> list[dict]:
    dataset = path.parent.name
    data = json.loads(path.read_text(encoding="utf-8"))
    rows = []
    for model_name, metrics in data.items():
        row = {
            "dataset": dataset,
            "model": model_name,
            "acc": None,
            "auroc": None,
            "brier": None,
            "ece": None,
        }
        if isinstance(metrics, dict):
            row.update(_canonical_metrics(metrics))
        rows.append(row)
    return rows


def collect_metrics(results_dir: Path) -> pd.DataFrame:
    rows: list[dict] = []
    for metrics_path in sorted(results_dir.glob("*/metrics.json")):
        rows.extend(_read_metrics_file(metrics_path))
    if not rows:
        return pd.DataFrame(columns=CANON_COLS)

    df = pd.DataFrame(rows)
    for col in CANON_COLS:
        if col not in df.columns:
            df[col] = None
    df = df[CANON_COLS]
    df = df.sort_values(["dataset", "model"], kind="stable")
    return df


def to_wide(df: pd.DataFrame) -> pd.DataFrame:
    metric_cols = [c for c in df.columns if c not in {"dataset", "model"}]
    if not metric_cols:
        return df.copy()

    wide = df.pivot(index="model", columns="dataset", values=metric_cols)
    wide.columns = [f"{dataset}.{metric}" for metric, dataset in wide.columns]
    wide = wide.reset_index()
    wide = wide.sort_values(["model"], kind="stable")
    return wide


def write_outputs(
    df_long: pd.DataFrame,
    out_csv: Path | None,
    out_xlsx: Path | None,
    update_workbook: Path | None,
    sheet_prefix: str,
) -> None:
    df_wide = to_wide(df_long)

    if out_csv is not None:
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        df_long.to_csv(out_csv, index=False, encoding="utf-8-sig")

    if out_xlsx is not None:
        out_xlsx.parent.mkdir(parents=True, exist_ok=True)
        with pd.ExcelWriter(out_xlsx, engine="openpyxl", mode="w") as writer:
            df_long.to_excel(writer, sheet_name="long", index=False)
            df_wide.to_excel(writer, sheet_name="wide", index=False)

    if update_workbook is not None:
        update_workbook.parent.mkdir(parents=True, exist_ok=True)
        mode = "a" if update_workbook.exists() else "w"
        with pd.ExcelWriter(
            update_workbook,
            engine="openpyxl",
            mode=mode,
            if_sheet_exists="replace",
        ) as writer:
            df_long.to_excel(writer, sheet_name=f"{sheet_prefix}_long", index=False)
            df_wide.to_excel(writer, sheet_name=f"{sheet_prefix}_wide", index=False)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Collect RLCR eval metrics into CSV/XLSX for easy Excel usage."
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("results"),
        help="Directory that contains per-dataset subfolders with metrics.json.",
    )
    parser.add_argument(
        "--out-csv",
        type=Path,
        default=Path("results/metrics_summary.csv"),
        help="Output CSV path (long format).",
    )
    parser.add_argument(
        "--out-xlsx",
        type=Path,
        default=Path("results/metrics_summary.xlsx"),
        help="Output XLSX path (contains long + wide sheets).",
    )
    parser.add_argument(
        "--update-workbook",
        type=Path,
        default=None,
        help="Optional: update an existing workbook by writing new sheets.",
    )
    parser.add_argument(
        "--sheet-prefix",
        type=str,
        default="metrics_auto",
        help="Sheet name prefix when --update-workbook is used.",
    )
    args = parser.parse_args()

    df_long = collect_metrics(args.results_dir)
    write_outputs(
        df_long=df_long,
        out_csv=args.out_csv if args.out_csv else None,
        out_xlsx=args.out_xlsx if args.out_xlsx else None,
        update_workbook=args.update_workbook,
        sheet_prefix=args.sheet_prefix,
    )

    print(f"rows={len(df_long)} results_dir={args.results_dir}")
    if args.out_csv:
        print(f"wrote {args.out_csv}")
    if args.out_xlsx:
        print(f"wrote {args.out_xlsx}")
    if args.update_workbook:
        print(
            f"updated {args.update_workbook} sheets: {args.sheet_prefix}_long, {args.sheet_prefix}_wide"
        )


if __name__ == "__main__":
    main()
