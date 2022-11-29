#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2022-11-28 19:35:07
# @Author  : Chenghao Mou (mouchenghao@gmail.com)
import os
import random
import re

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import networkit as nk
import numpy as np
import pandas as pd
import scienceplots  # This is needed
import datasets
from datasets import load_from_disk
from datasets.utils import disable_progress_bar
from rich.console import Console
from scipy.integrate import quad as integrate
from tqdm import tqdm

disable_progress_bar()
datasets.logging.set_verbosity_error()
random.seed(42)
np.random.seed(42)
plt.style.use("science")
NON_ALPHA = re.compile("[^A-Za-z_0-9]")
console = Console()


def set_jaccard_similarity(a, b):
    return len(a.intersection(b)) / len(a.union(b))


def false_positive_probability(threshold, b, r):
    probability = lambda s: 1 - (1 - s ** float(r)) ** float(b)
    a, _ = integrate(probability, 0.0, threshold)
    return a


def false_negative_probability(threshold, b, r):
    probability = lambda s: 1 - (1 - (1 - s ** float(r)) ** float(b))
    a, _ = integrate(probability, threshold, 1.0)
    return a


def optimal_param(threshold, num_perm, false_positive_weight, false_negative_weight):
    """
    Compute the optimal `MinHashLSH` parameter that minimizes the weighted sum
    of probabilities of false positive and false negative.
    """
    min_error = float("inf")
    opt = (0, 0)
    for b in range(1, num_perm + 1):
        max_r = int(num_perm / b)
        for r in range(1, max_r + 1):
            fp = false_positive_probability(threshold, b, r)
            fn = false_negative_probability(threshold, b, r)
            error = fp * false_positive_weight + fn * false_negative_weight
            if error < min_error:
                min_error = error
                opt = (b, r)
    return (
        *opt,
        threshold,
        num_perm,
        false_positive_probability(threshold, opt[0], opt[1]),
        false_negative_probability(threshold, opt[0], opt[1]),
        false_positive_weight,
        false_negative_weight,
    )


def draw_brace(ax, yspan, xx, text):
    """From https://stackoverflow.com/questions/18386210/annotating-ranges-of-data-in-matplotlib"""
    ymin, ymax = yspan
    yspan = ymax - ymin
    ax_ymin, ax_ymax = ax.get_ylim()
    yax_span = ax_ymax - ax_ymin

    xmin, xmax = ax.get_xlim()
    xspan = xmax - xmin
    resolution = int(yspan / yax_span * 100) * 2 + 1  # guaranteed uneven
    beta = 100.0 / yax_span  # the higher this is, the smaller the radius

    y = np.linspace(ymin, ymax, resolution)
    y_half = y[: int(resolution / 2) + 1]
    x_half_brace = 1 / (1.0 + np.exp(-beta * (y_half - y_half[0]))) + 1 / (1.0 + np.exp(-beta * (y_half - y_half[-1])))
    x = np.concatenate((x_half_brace, x_half_brace[-2::-1]))
    x = xx + (0.05 * x - 0.01) * xspan  # adjust vertical position

    ax.autoscale(False)
    ax.plot(x, y, lw=0.5, color="black")
    ax.text(xx + 0.6, (ymax + ymin) / 2 - 0.02, text, ha="center", va="bottom")


def _directory_find(goal, root="."):
    for path, dirs, _ in os.walk(root):
        if goal in dirs:
            return os.path.join(path, goal)
    raise FileNotFoundError(f"Could not find {goal} in {root}")


def _file_find(goal, root="."):
    for path, _, files in os.walk(root):
        if goal in files:
            return os.path.join(path, goal)
    raise FileNotFoundError(f"Could not find {goal} in {root}")


if __name__ == "__main__":

    import typer

    def run(
        sample_size: int = typer.Option(3_000, help="Number of sample clusters"),
        results_dir: str = typer.Option("results/codeparrot/codeparrot-clean-valid/default", help="Path to results"),
        output: str = typer.Option(None, help="Output plot file"),
        alternative_graph: bool = typer.Option(False, help="Use alternative graph"),
    ):
        theoretical_results = [
            optimal_param(0.85, 256, 0.5, 0.5),
            optimal_param(0.8, 256, 0.5, 0.5),
            optimal_param(0.8, 256, 0.9, 0.1),
            optimal_param(0.75, 256, 0.5, 0.5),
            optimal_param(0.7, 256, 0.5, 0.5),
            optimal_param(0.65, 256, 0.5, 0.5),
            optimal_param(0.6, 256, 0.5, 0.5),
            optimal_param(0.6, 512, 0.5, 0.5),
        ]

        console.print(
            pd.DataFrame(
                theoretical_results,
                columns=["𝐛", "𝐫", "threshold", "num_perm", "FP", "FN", "𝛼", "𝛽"],
            )
        )

        N = sample_size
        thresholds = list(map(int, (d for d in os.listdir(results_dir) if d.isdigit())))
        scores = [[] for _ in thresholds]
        removed = [0 for _ in thresholds]
        pbar = tqdm(thresholds, desc="Iterating over thresholds")

        for pos, threshold in enumerate(pbar):
            BASE = f"{results_dir}/{threshold}"
            ds = load_from_disk(_directory_find("indexed", BASE))
            if not alternative_graph:
                g = nk.readGraph(_file_find("graph.networkit", BASE), nk.Format.NetworkitBinary)
            else:
                g = nk.readGraph(_file_find("alternative.networkit", BASE), nk.Format.NetworkitBinary)

            cc = nk.components.ConnectedComponents(g)
            cc.run()
            components = list(cc.getComponents())
            components = [c for c in components if len(c) > 1]
            M = len(components)
            removed[pos] += sum(map(len, components)) - M
            pbar.set_description(f"Threshold: {threshold}, #components: {M}, #removed: {sum(map(len, components)) - M}")

            for idx in tqdm(
                random.sample(range(M), min(N, M)),
                leave=False,
                desc="Iterating over components",
            ):
                component = components[idx]
                S = len(component)
                matrix = np.zeros((S, S))
                subset = ds.select(component, keep_in_memory=True)
                content = [
                    set(tokens)
                    for tokens in subset.map(
                        lambda x: {"tokens": {t for t in NON_ALPHA.split(x["content"]) if t}},
                        remove_columns=subset.column_names,
                        num_proc=min(os.cpu_count(), S),
                    )["tokens"]
                ]
                for i in tqdm(range(S), leave=False, desc="Computing Jaccard similarity"):
                    for j in range(i + 1, S):
                        matrix[i, j] = matrix[j, i] = set_jaccard_similarity(content[i], content[j])
                scores[pos].extend(matrix.max(axis=1))

        _, ax = plt.subplots(figsize=(10, 5))
        ax.violinplot(scores, quantiles=[[0.25, 0.5, 0.75] for _ in scores])
        ax.hlines(
            [x / 100 for x in thresholds],
            [i - 0.2 for i, x in enumerate(thresholds, 1)],
            [i + 0.2 for i, x in enumerate(thresholds, 1)],
            colors=["red"],
            linestyles="dashed",
        )
        for i, s in enumerate(scores):
            s = np.asarray(s)
            draw_brace(
                ax,
                [min(s) + 0.01, thresholds[i] / 100],
                i + 1.1,
                f"{len(s[s < (thresholds[i] / 100)])/len(s) * 100:.2f}\%",
            )
        ax.set_xlim(0, 7)
        positions = list(range(8))
        labels = ["0.55"] + [f"{x / 100}\n({len(scores[i])}|{removed[i]})" for i, x in enumerate(thresholds)]
        ax.xaxis.set_major_locator(ticker.FixedLocator(positions))
        ax.xaxis.set_major_formatter(ticker.FixedFormatter(labels))

        ax.set_title(f"Actual Jaccard Similarity (unigram) Distribution w/ {N} Clusters")
        ax.set_ylabel("Actual Jaccard Similarity (unigram)")
        ax.set_xlabel("Threshold\n(Samples|Duplicates Removed)")

        plt.savefig(output, dpi=300, format="png")

    typer.run(run)
