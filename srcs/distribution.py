import os
from collections import defaultdict
import matplotlib.pyplot as plt
import argparse


def get_args() -> tuple[argparse.Namespace, argparse.ArgumentParser]:
    parser = argparse.ArgumentParser(
        description="Visualize class distribution in a dataset"
    )
    parser.add_argument(
        "-src",
        type=str,
        default=None,
        metavar="N",
        help="Provide the dataset folder path to visualize the class distribution",
    )
    return parser.parse_args(), parser


def check_args(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    if args.src is None:
        parser.error("You must specify the dataset folder path")
    assert args.src, "You must provide the dataset folder path"
    assert os.path.exists(args.src), "The path does not exist"
    assert os.path.isdir(args.src), "The path must be a directory"


def count_files(folder_path: str) -> dict:
    direct_files = [
        f
        for f in os.listdir(folder_path)
        if os.path.isfile(os.path.join(folder_path, f))
    ]
    if direct_files:
        return {os.path.basename(folder_path): len(direct_files)}

    counts = {}
    for entry in sorted(os.listdir(folder_path)):
        full = os.path.join(folder_path, entry)
        if os.path.isdir(full):
            counts[entry] = len(
                [f for f in os.listdir(full) if os.path.isfile(os.path.join(full, f))]
            )
    return counts


def group_by_plant(counts: dict) -> defaultdict:
    groups = defaultdict(dict)
    for name, count in counts.items():
        plant = name.split("_")[0]
        groups[plant][name] = count
    return groups


def plot_distribution(title: str | None, class_counts: dict) -> None:
    labels = list(class_counts.keys())
    values = list(class_counts.values())
    colors = plt.cm.tab10.colors[: len(labels)]

    fig, (ax_pie, ax_bar) = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(title or None)

    ax_pie.pie(
        values,
        labels=None,
        colors=colors,
        autopct="%1.1f%%",
        startangle=90,
    )
    ax_pie.legend(labels, loc="upper left", fontsize=8)

    ax_bar.bar(labels, values, color=colors)
    ax_bar.set_xticks(range(len(labels)))
    ax_bar.set_xticklabels(labels, rotation=15, ha="right", fontsize=8)
    ax_bar.set_ylabel("Count")

    plt.tight_layout()
    plt.show()


def distribution(folder_path: str) -> None:
    counts = count_files(folder_path)

    if not counts:
        print(f"No files or subdirectories found in {folder_path}")
        return

    groups = group_by_plant(counts)

    if len(groups) == 1:
        plot_distribution(None, counts)
    else:
        for plant, class_counts in groups.items():
            title = f"{plant.lower()} class distribution"
            plot_distribution(title, class_counts)


def main():
    args, parser = get_args()
    try:
        check_args(args, parser)
        distribution(args.src)
    except AssertionError as error:
        print(AssertionError.__name__ + ":", error)


if __name__ == "__main__":
    main()
