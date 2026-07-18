import pandas as pd
from scipy.stats import wilcoxon
from scipy.stats import binomtest


def pairwise_wilcoxon_returns(csv_path_a: str, csv_path_b: str, alpha: float = 0.05) -> str:
    """ Verifies with the Wilcoxon Signed Rank test if the model B has statistically
    better returns that model A
    """
    df_a = pd.read_csv(csv_path_a).set_index("seed")
    df_b = pd.read_csv(csv_path_b).set_index("seed")

    # pairing the two dataframes
    df_paired = df_a.join(df_b, lsuffix="_A", rsuffix="_B", how = "inner")

    n_common = len(df_paired)
    n_only_a = len(df_a) - n_common
    n_only_b = len(df_b) - n_common

    if n_common == 0:
        raise ValueError("The two CSV files share no seeds: paired comparison is impossible.")
    
    if n_only_a > 0 or n_only_b > 0:
        print(
            f"[Warning] Seeds in common: {n_common} "
            f"(only in A: {n_only_a}, only in B: {n_only_b}). "
            f"Test will run on common seeds only."
        )


    print(f"\nModel A: {csv_path_a}")
    print(f"Model B: {csv_path_b}")
    print(f"Paired episodes: {n_common}")
    
    # Wilcoxon signed-rank test on returns

    values_returnA = df_paired["return_A"].to_numpy(dtype=float)
    values_returnB = df_paired["return_B"].to_numpy(dtype=float)

    mean_a = values_returnA.mean()
    mean_b = values_returnB.mean()

    # Sample std (Bessel's correction), consistent with what evaluation/dqn.py computes
    std_a = values_returnA.std(ddof=1) if n_common > 1 else 0.0
    std_b = values_returnB.std(ddof=1) if n_common > 1 else 0.0

    n_ties = int((values_returnA - values_returnB == 0).sum())
    if n_ties == n_common:
        print("\nMetric 'return': all differences are 0, test not applicable.")

    # two sided wilcoxon test
    stat_two, p_two = wilcoxon(values_returnA, values_returnB, alternative="two-sided")

    # one sided test: the model with the higher mean is tested as the greater one
    if mean_b >= mean_a:
        better_label = f"{csv_path_b} > {csv_path_a} \n: {mean_b:.2f} +_ {std_b:.2f} VS"
        better_label += f"{mean_a:.2f} +_ {std_a:.2f}"

        stat_one, p_one = wilcoxon(values_returnB, values_returnA, alternative="greater")
    else:
        better_label = f"{csv_path_a} > {csv_path_b} \n: {mean_a:.2f} +_ {std_a:.2f} VS"
        better_label += f"{mean_b:.2f} +_ {std_b:.2f}"

        stat_one, p_one = wilcoxon(values_returnA, values_returnB, alternative="greater")

    significant_two = p_two < alpha
    significant_one = p_one < alpha

    results = better_label + f"\n Two sided Wilcoxon-Signed Rank Test for Returns (alpha = {alpha}):"
    if significant_two:
        results += f" SIGNIFICANT. (p_two = {p_two:.4e})"
    else:
        results += f" NOT significant. (p_two = {p_two:.4e})"
    
    results += f"\n One-sides Wilcoxon-Signed Rank Test for Returns (alpha = {alpha}):"
    if significant_one:
        results += f" SIGNIFICANT. (p_one = {p_one: .4e})"
    else:
        results += f" NOT significant. (p_one = {p_one: .4e})"

    return results

def pairwise_mcnemar_winrate(csv_path_a: str, csv_path_b: str, alpha: float = 0.05) -> str:
    """ Verifies with the exact McNemar test whether one model has a statistically
    different win rate than the other."""

    # in un confronto pair-wise su N partite, guardando solo se la partità è vinta
    # o perssa, la differenza tra esito della partita B ed esito della partita A
    # può assumere solo tre valoriù
    # d = esito_B - esito_A
    # d = 0 -> concordi, entrambi hanno vinto
    # d = +1 -> ha vinto B, contiamo il numero totale in cui ha vinto B con la variabile c
    # d = -1 -> ha vinto A, contiamo il numero totale in cui ha vinto A con la variabile b
    
    # il test di pairwise calcola la probabilità (p-value) di osservare c vittorie di B
    # su b + c vittorie totali. se questa probabilità è pari a 0.5, allora non si può
    # rigettare la null hypotesis e quindi i due modelli sono equivalenti.
    # questo test è detto two-sided binomial test.

    df_a = pd.read_csv(csv_path_a).set_index("seed")
    df_b = pd.read_csv(csv_path_b).set_index("seed")

    df_paired = df_a.join(df_b, lsuffix="_A", rsuffix="_B", how="inner")

    n_common = len(df_paired)
    n_only_a = len(df_a) - n_common
    n_only_b = len(df_b) - n_common
    if n_common == 0:
        raise ValueError("The two CSV files share no seeds: paired comparison is impossible.")
    
    if n_only_a > 0 or n_only_b > 0:
        print(
            f"[Warning] Seeds in common: {n_common} "
            f"(only in A: {n_only_a}, only in B: {n_only_b}). "
            f"Test will run on common seeds only."
        )

    # 'won' in the CSV is stored as a boolean (True/False), cast to int for arithmetic
    won_a = df_paired["won_A"].astype(int).to_numpy()
    won_b = df_paired["won_B"].astype(int).to_numpy()

    win_rate_a = won_a.mean()
    win_rate_b = won_b.mean()

    std_dev_a = won_a.std()
    std_dev_b = won_b.std()

    # concordant pairs (both win or both lose), not used in the test
    concordant = int(((won_a == 1) & (won_b == 1)).sum())   # both win
    concordant_loss = int(((won_a == 0) & (won_b == 0)).sum())  # both lose

    # discordant pairs: these are the only ones that drive the McNemar test
    b = int(((won_a == 1) & (won_b == 0)).sum())  # only A wins
    c = int(((won_a == 0) & (won_b == 1)).sum())  # only B wins
    n_discordant = b + c

    results = "MCNEMAR TEST RESULTS: \n"
    results += f"  Concordant pairs (both win):  {concordant}\n"
    results += f"  Concordant pairs (both lose): {concordant_loss}\n"
    results += f"  Discordant pairs (only A wins): b = {b}\n"
    results += f"  Discordant pairs (only B wins): c = {c}\n"

    if win_rate_a > win_rate_b:
        results += f"{csv_path_a} > {csv_path_b} for win rate. \n"
        results += f"{win_rate_a:.2f} +_ {std_dev_a} VS {win_rate_b:.2f} +_ {std_dev_b:.2f} \n"
    elif win_rate_b > win_rate_a:
        results += f"{csv_path_b} > {csv_path_a} for win rate. \n"
        results += f"{win_rate_b:.2f} +_ {std_dev_b} VS {win_rate_a:.2f} +_ {std_dev_a:.2f} \n"
    else:
        results += f"Identical win rate. \n"
    if n_discordant == 0:
        results += "All pairs are concordant: the two models have identical win patterns. \n"
        return results
    
    res = binomtest(c, n_discordant, p=0.5, alternative="two-sided")
    p_value = res.pvalue
    significant = p_value < alpha
    if significant:
        results += f"SIGNIFICANT. p_value: {p_value:.4e} alpha: {alpha} \n"
    else:
        results += f"NOT significant. p_value: {p_value:.4e} alpha: {alpha} \n"
    
    return results