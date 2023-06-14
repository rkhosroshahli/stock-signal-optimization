import numpy as np
import pandas as pd
from tqdm import tqdm

cols = ['T', 'Buy date', 'Buy price', 'Sell date', 'Sell price', 'Profit Ratio']


def exhaustive_searcher(objective, n_bits, n_iter):
    # enumerate generations
    binary_format = f'0{n_bits}b'
    all_scores = np.zeros(n_iter)
    all_bin_sols = np.zeros((n_iter, n_bits))
    for i in tqdm(range(n_iter)):
        sol = np.array(list(format(i, binary_format))).astype(int)
        all_bin_sols[i] = sol
        # evaluate all candidates in the population
        obj_val = objective(sol, 0, i)
        all_scores[i] = obj_val
        # save it to output
        # new_row = ['Sum', '', '', '', '', obj_val]
        # new_df = pd.DataFrame([new_row], columns=cols)
        # df = pd.concat([df, new_df], ignore_index=True)
        # df.to_csv(f"./outputs/sol{i}_transactions.csv", index=False)

    return all_scores, all_bin_sols
