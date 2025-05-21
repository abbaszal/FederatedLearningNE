import pandas as pd
import re


def remove_redundant_information(df_results: pd.DataFrame) -> pd.DataFrame:
    """
    NOTE: No need to have "Client i Accuracy" columns. The same information can be retrieved either from:
    - the global accuracy for the one-hot coalition #i `(0...1...0)`, if the i-th client does not join the current coalition
    - the global accuracy of the current coalition, if the i-th client joins the current coalition
    """
    redundant_cols = [col for col in df_results.columns if re.match(r"Client \d+ Accuracy", col) is not None]
    df_results = df_results.drop(columns=redundant_cols)
    return df_results


def one_hot_coalition(player: int, n_players: int) -> str:
    """
    Args:
        player (int): Player index (between `1` and `n_players`).
    
    Return:
        one_hot (str): one-hot coalition where only `player` is participating.
                      Note: The coalition string is defined so that the right-most bit corresponds to player 1,
                      and the left-most bit corresponds to player n.
    """
    one_hot = ["0"] * n_players
    one_hot[n_players - player] = "1"  # right-to-left: player 1 is at index n_players-1
    one_hot = "".join(one_hot)
    return one_hot


def evaluate_deviation(df_results: pd.DataFrame, coalition: str, player: int) -> bool:
    """
    Args:
        df_results (pd.DataFrame): DataFrame containing the results (should be indexed by the coalition)
        coalition (str): binary string representing the coalition.
        player (int): Player whose deviation should be evaluated (index between `1` and `n_players`).

    Return:
        wants_to_deviate (bool): True if player has incentive to deviate, otherwise False.
    
    Note:
        The coalition string is interpreted from right-to-left.
        Therefore, the bit for player 1 is the right-most bit.
    """
    # Access the player's move using right-to-left indexing.
    current_move = coalition[-player]
    
    # Get the standalone (one-hot) coalition for this player.
    one_hot_player = one_hot_coalition(player, len(coalition))
    standalone_payoff = df_results.loc[one_hot_player, "Global Accuracy"]
    
    if current_move == "0":
        # Player is not in the coalition.
        # Check if joining (i.e., setting their bit to "1") would yield a higher payoff.
        new_coalition = list(coalition)
        new_coalition[-player] = "1"
        new_coalition = "".join(new_coalition)
        coalition_payoff = df_results.loc[new_coalition, "Global Accuracy"]
        return coalition_payoff + 0.000001 > standalone_payoff

    elif current_move == "1":
        # Player is in the coalition.
        # Check if leaving (i.e., using standalone payoff) would yield a better outcome.
        coalition_payoff = df_results.loc[coalition, "Global Accuracy"]
        return standalone_payoff + 0.000001 > coalition_payoff

    else:
        raise ValueError("This should not happen")


def find_nash_equilibria_v2(df_results: pd.DataFrame):
    df_results = df_results.copy()
    df_results = remove_redundant_information(df_results)

    df_results['Combination'] = df_results['Combination'].astype(str)
    n_clients = max(df_results['Combination'].apply(lambda x: len(x)))
    df_results['Combination'] = df_results['Combination'].apply(lambda x: x.zfill(n_clients))

    df_results = df_results.set_index('Combination')

    nash_equilibria = []

    for coalition in df_results.index:
        is_nash = True
        for player in range(1, n_clients + 1):
            if evaluate_deviation(df_results, coalition, player):
                is_nash = False
                break
        if is_nash:
            nash_equilibria.append(coalition)

    df_ne = df_results.loc[nash_equilibria]
    df_ne = df_ne.reset_index(drop=False)

    return df_ne




