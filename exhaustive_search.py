import numpy as np
import pandas as pd
from tqdm import tqdm

cols = ['T', 'Buy date', 'Buy price', 'Sell date', 'Sell price', 'Profit Ratio']


def exhaustive_searcher(objective, n_bits):
    # enumerate generations
    binary_format = f'0{n_bits}b'
    n_iter = 2 ** n_bits
    all_scores = np.zeros(n_iter)
    all_bin_sols = np.zeros((n_iter, n_bits))
    for i in tqdm(range(n_iter)):
        sol = np.array(list(format(i, binary_format))).astype(int)
        all_bin_sols[i] = sol
        # evaluate all candidates in the population
        obj_val, df = objective(sol, 0, i)
        all_scores[i] = obj_val
        # save it to output
        # new_row = ['Sum', '', '', '', '', obj_val]
        # new_df = pd.DataFrame([new_row], columns=cols)
        # df = pd.concat([df, new_df], ignore_index=True)
        # df.to_csv(f"./outputs/sol{i}_transactions.csv", index=False)
        # np.save(f"./outputs/sol{i}_rules", sol)

    sorted_scores_idx = np.argsort(all_scores)[::-1]
    sorted_scores = all_scores[sorted_scores_idx]

    binary_format = f'0{n_bits}b'
    df = pd.DataFrame()
    for i in range(n_iter):
        bin_sol = all_bin_sols[i]
        hold_days = 0
        for ele in bin_sol[-4:].astype(dtype=int):
            hold_days = (hold_days << 1) | ele
        hold_days += 1
        df[f"Solution {i + 1}"] = np.concatenate([bin_sol, [hold_days], [f"{sorted_scores[i]:.6f}"]])

    return df


def get_solution_dataframe(objective, i):
    sol = np.load(f"./outputs/sol{i}_rules")
    obj_val, df = objective(sol, 0, i)
    # save it to output
    new_row = ['Sum', '', '', '', '', obj_val]
    new_df = pd.DataFrame([new_row], columns=cols)
    df = pd.concat([df, new_df], ignore_index=True)
    df.to_csv(f"./outputs/sol{i}_transactions.csv", index=False)

    return df
