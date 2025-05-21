import pandas as pd
import re
import numpy as np
from typing import Callable

def one_hot_coalition(player: int, n_players: int) -> str:
    """One-hot bitstring of length n_players with only `player` set to '1'."""
    bits = ["0"] * n_players
    bits[player - 1] = "1"
    return "".join(bits)

def evaluate_deviation_lottery(
    df: pd.DataFrame,
    coalition: str,
    player: int
) -> bool:
    """
    Return True if `player` has a profitable unilateral deviation:
      - Solo payoff = that client’s *own* Accuracy column.
      - Coalition payoff = df["Predicted Global Accuracy"].
    """
    n = len(coalition)
    solo = one_hot_coalition(player, n)
    local_col = f"Client {player} Accuracy"
    solo_payoff = df.at[solo, local_col]


    if coalition[player - 1] == "0":
        # test joining
        joined = coalition[:player-1] + "1" + coalition[player:]
        joined_payoff = df.at[joined, "Predicted Global Accuracy"]
        deviable = (joined_payoff + 1e-6 > solo_payoff)
        print(f"[Join] Player {player}: solo={solo_payoff:.4f}, "
              f"joined={joined_payoff:.4f} → {deviable}")
        return joined_payoff + 1e-6 > solo_payoff 
    else:
        # test leaving (going solo)
        coalition_payoff = df.at[coalition, "Predicted Global Accuracy"]
        deviable = (solo_payoff + 1e-6 > coalition_payoff)
        print(f"[Leave] Player {player}: solo={solo_payoff:.4f}, "
              f"coalition={coalition_payoff:.4f} → {deviable}")
        return solo_payoff + 1e-6 > coalition_payoff

def find_nash_equilibria_lottery(
    df_results: pd.DataFrame,
    payoff_func: Callable[[float, float], float]
) -> pd.DataFrame:
    """
    Returns a DataFrame of all pure-strategy Nash equilibria,
    INCLUDING each client’s local accuracy and the global payoff.
    """
    # 1) normalize 'Combination' to n-bit strings, set as index
    df = df_results.copy()
    df['Combination'] = df['Combination'].astype(str)
    n_clients = df['Combination'].str.len().max()
    df['Combination'] = df['Combination'].str.zfill(n_clients)
    df = df.set_index('Combination')

    # 2) find all local-accuracy columns
    local_cols = [c for c in df.columns if re.match(r"Client \d+ Accuracy", c)]

    # 3) compute μ, σ, and global payoff for *every* coalition
    mus, sigmas, preds = [], [], []
    for combo, row in df.iterrows():
        bits = np.array(list(combo), dtype=int)
        accs = row[local_cols].astype(float).values
        included = accs[bits == 1]
        mu    = included.mean()   if included.size else 0.0
        sigma = included.std(ddof=0) if included.size else 0.0
        Ag    = payoff_func(mu, sigma)
        mus.append(mu)
        sigmas.append(sigma)
        preds.append(Ag)

    df['mu']                       = mus
    df['sigma']                    = sigmas
    df['Predicted Global Accuracy'] = preds

    # 4) find all coalitions with *no* profitable single-player deviation
    nash_coalitions = []
    for coalition in df.index:
        if not any(
            evaluate_deviation_lottery(df, coalition, p)
            for p in range(1, n_clients + 1)
        ):
            nash_coalitions.append(coalition)

    # 5) build result: reset index and select exactly the columns you want
    result = df.loc[nash_coalitions].reset_index()
    # include each client’s accuracy plus mu, sigma, Predicted Global Accuracy
    cols = ['Combination'] + local_cols + ['mu', 'sigma', 'Predicted Global Accuracy']
    return result[cols]


