"""Export main-method tables and figures from the authoritative result JSON."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from paper_benchmarks import DEFAULT_BENCHMARK_ROOT, PAPER_CASE_NAMES, PAPER_SOURCE_LABELS, load_paper_examples
from paper_pipeline import best_candidate


def _font(size: int, bold: bool = False):
    candidates = [
        "C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf",
        "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf",
    ]
    for candidate in candidates:
        try:
            return ImageFont.truetype(candidate, size)
        except OSError:
            continue
    return ImageFont.load_default()


def _case_rows(rows: list[dict[str, Any]], case_id: str) -> list[dict[str, Any]]:
    return sorted(
        [row for row in rows if str(row["example"]) == case_id],
        key=lambda row: int(row["barrier_degree"]),
    )


def _write_table(rows: list[dict[str, Any]], path: Path) -> None:
    lines: list[str] = []
    for index in range(1, 10):
        case_id = f"C{index}"
        candidates = _case_rows(rows, case_id)
        best = best_candidate(candidates)
        source = PAPER_SOURCE_LABELS.get(case_id, str(candidates[0].get("source_case", case_id))) if candidates else "--"
        display_name = PAPER_CASE_NAMES.get(case_id, case_id)
        if best is None:
            lines.append(f"{display_name} ({source}) & -- & 0 & -- & not verified \\\\")
        else:
            lines.append(
                f"{display_name} ({source}) & {int(best['barrier_degree'])} & "
                f"{float(best['safety_lower_bound']):.6f} & "
                f"{float(best['solve_time_seconds']):.2f} & verified \\\\")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _latex_term(coefficient: float, exponent: list[int], first: bool) -> str:
    if abs(coefficient) < 5e-13:
        return ""
    sign = "-" if coefficient < 0 else ("" if first else "+")
    factors = []
    for index, power in enumerate(exponent, start=1):
        if int(power) == 1:
            factors.append(f"x_{{{index}}}")
        elif int(power) > 1:
            factors.append(f"x_{{{index}}}^{{{int(power)}}}")
    monomial = " ".join(factors)
    magnitude = f"{abs(coefficient):.8g}"
    return f"{sign}{magnitude}{(' ' + monomial) if monomial else ''}"


def _write_case_certificate(rows: list[dict[str, Any]], case_id: str, path: Path) -> None:
    best = best_candidate(_case_rows(rows, case_id))
    if best is None:
        path.write_text(f"% No verified certificate for {case_id}.\n", encoding="utf-8")
        return
    terms: list[str] = []
    for coefficient, exponent in zip(
        best["barrier_coefficients"], best["barrier_exponents"]
    ):
        term = _latex_term(float(coefficient), list(exponent), not terms)
        if term:
            terms.append(term)
    wrapped = " ".join(terms)
    text = (
        "\\begin{align}\n"
        f"B_{{{case_id}}}(x) ={{}}& {wrapped},\\\\\n"
        f"\\rho_{{{case_id}}} ={{}}& {float(best['reported_rho']):.9g}.\n"
        "\\end{align}\n"
    )
    path.write_text(text, encoding="utf-8")


def _plot_benchmark(rows: list[dict[str, Any]], path: Path) -> None:
    width, height = 1800, 980
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    draw.text(
        (width // 2, 45),
        "Verified safety lower bounds for the proposed method",
        fill="#202020",
        font=_font(40, True),
        anchor="ma",
    )
    plot = (150, 135, 1740, 825)
    draw.line((plot[0], plot[3], plot[2], plot[3]), fill="#222222", width=3)
    draw.line((plot[0], plot[1], plot[0], plot[3]), fill="#222222", width=3)
    for tick in range(6):
        value = tick / 5
        y = int(plot[3] - value * (plot[3] - plot[1]))
        draw.line((plot[0], y, plot[2], y), fill="#dddddd", width=2)
        draw.text((plot[0] - 20, y), f"{value:.1f}", fill="#333333", font=_font(25), anchor="rm")
    slot = (plot[2] - plot[0]) / 9
    for index in range(9):
        case_id = f"C{index + 1}"
        display_name = PAPER_CASE_NAMES.get(case_id, case_id)
        best = best_candidate(_case_rows(rows, case_id))
        value = 0.0 if best is None else float(best["safety_lower_bound"])
        left = int(plot[0] + index * slot + 0.2 * slot)
        right = int(plot[0] + (index + 1) * slot - 0.2 * slot)
        top = int(plot[3] - value * (plot[3] - plot[1]))
        draw.rectangle((left, top, right, plot[3]), fill="#2878b5", outline="#164d73", width=2)
        draw.text(((left + right) // 2, plot[3] + 20), display_name, fill="#222222", font=_font(24, True), anchor="ma")
        draw.text(((left + right) // 2, max(plot[1] + 10, top - 12)), f"{value:.3f}", fill="#222222", font=_font(22), anchor="ms")
    draw.text((45, (plot[1] + plot[3]) // 2), "Safety lower bound", fill="#222222", font=_font(29, True), anchor="mm")
    image.save(path, dpi=(220, 220), optimize=True)


def _plot_degree_sweep(rows: list[dict[str, Any]], case_id: str, path: Path) -> None:
    candidates = _case_rows(rows, case_id)
    width, height = 1300, 820
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    draw.text((width // 2, 40), f"{PAPER_CASE_NAMES.get(case_id, case_id)} barrier-degree sweep", fill="#202020", font=_font(38, True), anchor="ma")
    plot = (140, 125, 1220, 680)
    draw.line((plot[0], plot[3], plot[2], plot[3]), fill="#222222", width=3)
    draw.line((plot[0], plot[1], plot[0], plot[3]), fill="#222222", width=3)
    for tick in range(6):
        value = tick / 5
        y = int(plot[3] - value * (plot[3] - plot[1]))
        draw.line((plot[0], y, plot[2], y), fill="#dddddd", width=2)
        draw.text((plot[0] - 15, y), f"{value:.1f}", fill="#333333", font=_font(23), anchor="rm")
    points = []
    for index, degree in enumerate((2, 4, 6)):
        row = next((item for item in candidates if int(item["barrier_degree"]) == degree), None)
        value = 0.0 if row is None else float(row["safety_lower_bound"])
        x = int(plot[0] + (index + 0.5) * (plot[2] - plot[0]) / 3)
        y = int(plot[3] - value * (plot[3] - plot[1]))
        points.append((x, y))
        draw.text((x, plot[3] + 20), str(degree), fill="#222222", font=_font(27, True), anchor="ma")
        draw.text((x, y - 18), f"{value:.4f}", fill="#222222", font=_font(22), anchor="ms")
    draw.line(points, fill="#2878b5", width=6)
    for point in points:
        draw.ellipse((point[0] - 10, point[1] - 10, point[0] + 10, point[1] + 10), fill="#2878b5", outline="#164d73", width=3)
    draw.text((680, 770), "Barrier degree", fill="#222222", font=_font(28, True), anchor="mm")
    image.save(path, dpi=(220, 220), optimize=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--case", default="C6")
    parser.add_argument("--benchmark-root", type=Path, default=DEFAULT_BENCHMARK_ROOT)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    payload = json.loads(args.results.read_text(encoding="utf-8"))
    rows = list(payload["candidates"])
    load_paper_examples(args.benchmark_root)
    case_id = str(args.case).upper()
    args.out.mkdir(parents=True, exist_ok=True)
    _write_table(rows, args.out / "paper_main_table_rows.tex")
    _write_case_certificate(rows, case_id, args.out / f"paper_main_{case_id.lower()}_sbc.tex")
    _plot_benchmark(rows, args.out / "paper_main_benchmark_summary.png")
    _plot_degree_sweep(rows, case_id, args.out / f"paper_main_{case_id.lower()}_degree_sweep.png")
    print(f"paper artifacts: {args.out.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
