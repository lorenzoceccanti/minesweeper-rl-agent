from pathlib import Path


def _short_name(csv_path: str) -> str:
    """Derives a shortened name starting from a fully-qualified CSV file name"""
    # Example: csv/dqn/dqn_fully-conv_9x9@10_seed43_results.csv
    # -> dqn_fully-conv_9x9@10_seed43
    stem = Path(csv_path).stem          # es. dqn_fully-conv_9x9@10_seed43_results
    return stem.removesuffix("_results")  # es. dqn_fully-conv_9x9@10_seed43


def write_txt(output: str, folder_path: str, csv_a: str, csv_b: str) -> None:
    """Writes *output* in <folder_path>/<nameA>-VS-<nameB>.txt.
    """
    # names are derived from stems of CSV files already in config.yaml
    name_a = _short_name(csv_a)   # es. dqn_fully-conv_9x9@10_seed43
    name_b = _short_name(csv_b)   # es. ppo_fully-conv_9x9@10_seed43

    dest = Path(folder_path)
    dest.mkdir(parents=True, exist_ok=True)   # crea la cartella se non esiste

    file_path = dest / f"{name_a}-VS-{name_b}.txt"

    with open(file_path, "w", encoding="utf-8") as fh:
        fh.write(output)
    print(f"[stats] results saved in: {file_path.resolve()}")