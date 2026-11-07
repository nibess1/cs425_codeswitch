#!/usr/bin/env python3
"""Human stress set (7 Nov): figure 8 -> results/figures/fig8_human_quant_delta.{png,svg,csv}.

Reads only results/tables/27b_human_fragility_R.csv (run make_human_tables.py first); same style as make_figures.py.
  python scripts/make_human_figure.py
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from make_figures import forest, save, tab  # noqa: E402

MODELS = ["ft_tiny", "ft_base", "full_base", "ft_small", "full_small"]
FMT = {"8-bit LLM.int8 (GPU, simulated)": "LLM.int8", "4-bit NF4 (GPU, simulated)": "NF4",
       "CTranslate2 int8 (CPU)": "CT2 int8", "whisper.cpp Q8_0 (CPU)": "Q8_0", "whisper.cpp Q4_0 (CPU)": "Q4_0"}


def main():
    d = tab("27b_human_fragility_R.csv")
    d = d[(d.kind == "R") & (d["main"] == "all")].copy()
    d["format"] = d.precision.map(FMT)
    d["order"] = d.model.map(MODELS.index) * 10 + d.format.map(list(FMT.values()).index)
    d = d.sort_values("order")
    d["label"] = d.model.str.replace("_", " ") + ", " + d.format
    cols = ["label", "system_id", "d_embedded_points", "d_embedded_ci95_low", "d_embedded_ci95_high",
            "d_matrix_points", "d_matrix_ci95_low", "d_matrix_ci95_high"]
    d = d[cols].reset_index(drop=True)
    fig, axs = plt.subplots(1, 2, figsize=(9, 6.5), sharey=True)
    forest(axs[0], d, "label", "d_embedded_points", "d_embedded_ci95_low", "d_embedded_ci95_high",
           xlabel="change in embedded error (points)")
    forest(axs[1], d, "label", "d_matrix_points", "d_matrix_ci95_low", "d_matrix_ci95_high",
           xlabel="change in matrix error (points)")
    axs[0].set_title("Embedded units (E)", fontsize=10)
    axs[1].set_title("Matrix units (code-switched utterances)", fontsize=10)
    axs[1].legend(loc="lower right", fontsize=7)
    save(fig, d, "fig8_human_quant_delta")


if __name__ == "__main__":
    main()
