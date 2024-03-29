import numpy as np
import pandas as pd

cols = ['T', 'Buy date', 'Buy price', 'Sell date', 'Sell price', 'Profit Ratio']


def fitness_func(sol, sol_idx, generation_counter):
    sol = np.array(sol)

    df = pd.DataFrame(columns=cols)

    all_profit = []
    current_day, buy_day, sell_day = 0, 0, 0
    conds = np.zeros(sol.shape[0] - 1)
    trans_counter = 0
    hold_days = 0
    for ele in sol[-4:].astype(dtype=int):
        hold_days = (hold_days << 1) | ele

    # print(hold_days, sol[-4:])
    hold_days += 1

    counter = 0
    while current_day < df1.shape[0]:
        counter += 1
        if current_day + hold_days >= df1.shape[0]:
            break

        conds = all_rules[current_day].astype(dtype=int) ^ sol[:-4].astype(dtype=int)
        if sum(conds) == 0:
            if buy_day == 0:  # if the buy signal is seen and buy hasn't been initialized
                trans_counter += 1
                buy_day = current_day
                buy_price = df1.iloc[buy_day + 1]['Open']
                buy_date = df1.iloc[buy_day + 1].name.strftime("%m/%d/%Y")

                current_day = int(hold_days + buy_day)
                sell_day = current_day
                sell_price = df1.iloc[sell_day]['Close']
                sell_date = df1.iloc[sell_day].name.strftime("%m/%d/%Y")

                profit = (sell_price - buy_price) / buy_price
                all_profit.append(profit)
                new_row = [f'T{trans_counter}', buy_date, buy_price, sell_date, sell_price, profit]
                new_df = pd.DataFrame([new_row], columns=cols)
                df = pd.concat([df, new_df], ignore_index=True)
                buy_day = 0
        else:
            current_day += 1
        conds = np.zeros(sol.shape[0] - 1)

    if len(all_profit) > 0:
        fitness_val = np.mean(all_profit)
        """if fitness_val > best_gen_fitness_val:
            new_row=[f'Sum', '', '', '', '', fitness_val]
            new_df=pd.DataFrame([new_row], columns=cols)
            df=pd.concat([df, new_df], ignore_index=True)
            df.to_csv(f"./outputs/gen{generation_counter}_sol{sol_idx}_transactions.csv", index=False)
            np.save(f"./outputs/gen{generation_counter}_sol{sol_idx}_rules", sol)
            best_gen_fitness_val=fitness_val"""
        return fitness_val, df
    else:
        return 0.0, df