"""Aggregates the judgements of judge_pairwise.py into win rates.

Wins, ties and losses of both orderings of a model pair are pooled, and ties count as half a win
(the 1/0.5/0 aggregation of the paper). Prints a win-rate matrix with the average win rate of every
model and, with --baseline, the win rate of every model against the baseline (the NLG score).
"""
import argparse
import json
import os
from collections import defaultdict
from statistics import mean

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def parse_judgement(text: str):
    """The judge ends its message with A, B or tie."""
    tokens = text.strip().split()
    verdict = tokens[-1].lower().strip(".*") if tokens else ""
    return verdict if verdict in ("a", "b", "tie") else None


def load_colormap(path):
    coolwarm = []
    with open(path) as f:
        for i, line in enumerate(f):
            if i == 0:
                continue
            coolwarm.append([float(n) for n in line.strip().split(",")[1:]])
    return coolwarm


def rgb_from_winrate(winrate, coolwarm):
    winrate = 1.0 - winrate  # high win rates are blue
    winrate = (winrate - 0.5) * 0.8 + 0.5
    return coolwarm[min(len(coolwarm) - 1, int(winrate * 1024 + 0.5))]


def generate_latex_table(models, winrates, average_winrates, colormap_path):
    coolwarm = load_colormap(colormap_path)
    latex = [
        "\\begin{table}",
        "\\small",
        "\\begin{tabular}{@{}l" + "c" * len(models) + "@{\\hspace{2em}}c@{}}",
        "\\toprule",
        "Model & " + " & ".join(f"\\rotatebox{{90}}{{{m}}}" for m in models) + " & Avg. \\\\",
        "\\midrule",
    ]
    for model_row in models:
        row = [model_row]
        for model_col in models:
            if model_row == model_col:
                row.append("--")
            else:
                r, g, b = rgb_from_winrate(winrates[model_row][model_col], coolwarm)
                row.append(f"\\cellcolor[rgb]{{{r:.2f},{g:.2f},{b:.2f}}}{winrates[model_row][model_col] * 100:.1f}")
        r, g, b = rgb_from_winrate(average_winrates[model_row], coolwarm)
        row.append(f"\\cellcolor[rgb]{{{r:.2f},{g:.2f},{b:.2f}}}{average_winrates[model_row] * 100:.1f}\\%")
        latex.append(" & ".join(row) + " \\\\")
    latex += ["\\bottomrule", "\\end{tabular}", "\\caption{Model win rates}", "\\end{table}"]
    return "\n".join(latex)


def main():
    parser = argparse.ArgumentParser(description="Compute win rates from pairwise judgements.")
    parser.add_argument("--input", type=str, required=True, help="JSON file produced by judge_pairwise.py")
    parser.add_argument("--output", type=str, default=None, help="Text file with the win-rate matrix; defaults to {input}_winrates.txt")
    parser.add_argument("--baseline", type=str, default=None, help="Also print the win rate of every model against this model.")
    parser.add_argument("--latex", action="store_true", help="Print a colored LaTeX table (needs the xcolor and booktabs packages).")
    parser.add_argument("--colormap", type=str, default=os.path.join(SCRIPT_DIR, "..", "paper", "smooth-cool-warm-table-float-1024.csv"))
    args = parser.parse_args()
    if args.output is None:
        args.output = os.path.splitext(args.input)[0] + "_winrates.txt"

    with open(args.input, encoding="utf-8") as f:
        comparisons = json.load(f)["results"]

    models = sorted({model for key in comparisons for model in key.split("|")})
    counts = {a: {b: defaultdict(int) for b in models if b != a} for a in models}

    n_prefers_a, n_prefers_b, n_faulty = 0, 0, 0
    total_length, total_win_length, total_reference_length, n_total, n_win_total = 0, 0, 0, 0, 0
    for key, results in comparisons.items():
        model_1, model_2 = key.split("|")
        for result in results:
            judgement = parse_judgement(result["judgement"])
            total_length += len(result["prediction_1"].split())
            total_reference_length += len(result["reference_output"].split())
            n_total += 1

            if judgement == "a":
                counts[model_1][model_2]["wins"] += 1
                n_prefers_a += 1
                total_win_length += len(result["prediction_1"].split())
                n_win_total += 1
            elif judgement == "b":
                counts[model_1][model_2]["losses"] += 1
                n_prefers_b += 1
                total_win_length += len(result["prediction_2"].split())
                n_win_total += 1
            elif judgement == "tie":
                counts[model_1][model_2]["ties"] += 1
            else:
                n_faulty += 1

    print("\nPOSITION BIAS")
    print(f"Judge prefers response A {n_prefers_a} times")
    print(f"Judge prefers response B {n_prefers_b} times")
    print(f"Unparsable judgements: {n_faulty}")
    print("\nLENGTH BIAS")
    print(f"Average response length: {total_length / max(1, n_total):.2f}")
    print(f"Average reference length: {total_reference_length / max(1, n_total):.2f}")
    print(f"Average winning response length: {total_win_length / max(1, n_win_total):.2f}\n")

    # pool both orderings of every pair; ties count as half a win
    winrates = {a: {} for a in models}
    for i, model_1 in enumerate(models):
        for model_2 in models[i + 1:]:
            wins = counts[model_1][model_2]["wins"] + counts[model_2][model_1]["losses"]
            losses = counts[model_1][model_2]["losses"] + counts[model_2][model_1]["wins"]
            ties = counts[model_1][model_2]["ties"] + counts[model_2][model_1]["ties"]
            n = wins + losses + ties
            if n == 0:
                continue
            winrates[model_1][model_2] = (wins + ties / 2) / n
            winrates[model_2][model_1] = (losses + ties / 2) / n

    average_winrates = {model: mean(winrates[model].values()) for model in models if winrates[model]}

    lines = ["model\t" + "\t".join(models)]
    for model in models:
        lines.append(model + "\t" + "\t".join(f"{winrates[model][m]:.1%}" if m in winrates[model] else "-----" for m in models))
    lines.append("\naverage\t" + "\t".join(f"{average_winrates[m]:.1%}" if m in average_winrates else "-----" for m in models))
    table = "\n".join(lines) + "\n"
    print(table)
    with open(args.output, "w") as f:
        f.write(table)

    if args.baseline is not None:
        assert args.baseline in models, f"Unknown model {args.baseline!r}, choose from {models}"
        print(f"WIN RATES AGAINST {args.baseline}")
        for model in models:
            if model != args.baseline and args.baseline in winrates[model]:
                print(f"{model}\t{winrates[model][args.baseline]:.1%}")
        print()

    if args.latex:
        print(generate_latex_table(models, winrates, average_winrates, args.colormap))


if __name__ == "__main__":
    main()
