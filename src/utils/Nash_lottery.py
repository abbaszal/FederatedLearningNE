import pandas as pd
import re
import numpy as np
from typing import Callable

def one_hot_coalition(player: int, n_players: int) -> str:
    """One-hot string with only `player` set to '1'."""
    bits = ["0"] * n_players
    bits[player-1] = "1"
    return "".join(bits)

def evaluate_deviation_lottery(df: pd.DataFrame, coalition: str, player: int) -> bool:
    """
    Return True if `player` can improve payoff by deviating from `coalition` in the lottery game.
    Assumes df is indexed by coalition strings and contains 'Predicted Global Accuracy'.
    """
    n = len(coalition)
    solo = one_hot_coalition(player, n)
    local_col = f"Client {player} Accuracy"
    solo_payoff = df.at[solo, local_col]

    if coalition[player-1] == "0":
        # incentive to join?
        joined = list(coalition)
        joined[player-1] = "1"
        joined = "".join(joined)
        return df.at[joined, "Predicted Global Accuracy"] + 1e-6 > solo_payoff 
    else:
        # incentive to leave?
        coalition_payoff = df.at[coalition, "Predicted Global Accuracy"]
        return solo_payoff  + 1e-6 > coalition_payoff 


def find_nash_equilibria_lottery(
    df_results: pd.DataFrame,
    payoff_func: Callable[[float, float], float]
) -> pd.DataFrame:
    df = df_results.copy()
    df['Combination'] = df['Combination'].astype(str)
    n_clients = df['Combination'].str.len().max()
    df['Combination'] = df['Combination'].str.zfill(n_clients)
    df = df.set_index('Combination')

    # Identify local accuracy columns
    local_cols = [c for c in df.columns if re.match(r"Client \d+ Accuracy", c)]

    # Compute mu, sigma, Predicted Global Accuracy
    mus, sigmas, preds = [], [], []
    for combo, row in df.iterrows():
        bits = np.array(list(combo), dtype=int)
        accs = row[local_cols].values.astype(float)
        included = accs[bits == 1]
        mu = included.mean() if included.size else 0.0
        sigma = included.std(ddof=0) if included.size else 0.0
        Ag = payoff_func(mu, sigma)
        mus.append(mu)
        sigmas.append(sigma)
        preds.append(Ag)

    df['mu'] = mus
    df['sigma'] = sigmas
    df['Predicted Global Accuracy'] = preds

    # Determine Nash equilibria
    nash_coalitions = []
    for coalition in df.index:
        # no profitable deviation for any player
        if not any(evaluate_deviation_lottery(df, coalition, p)
                   for p in range(1, n_clients+1)):
            nash_coalitions.append(coalition)

    # Return only the Nash rows
    return df.loc[nash_coalitions].reset_index()
