import os
from exhaustive_search import exhaustive_searcher
from stock_processor import stock_processor
import numpy as np
import pandas as pd

cols = ['T', 'Buy date', 'Buy price', 'Sell date', 'Sell price', 'Profit Ratio']


def stock_fitness(sol, sol_idx, generation_counter):
    sol = np.array(sol)

    tickers_profits = np.zeros(shape=len(tickers_list))

    hold_days = 0
    for ele in sol[-4:].astype(dtype=int):
        hold_days = (hold_days << 1) | ele
    hold_days += 1
    for i in range(len(tickers_list)):
        t_rule = tickers_rules[i]
        t_dataframe = tickers_dataframes[i]
        conds = np.zeros(sol.shape[0] - 1)
        trans_counter = 0
        sum_profits = 0
        counter = 0
        current_day, buy_day, sell_day = 0, 0, 0
        while current_day < t_rule.shape[0]:
            if current_day + hold_days >= t_rule.shape[0]:
                break

            conds = t_rule[current_day].astype(dtype=int) ^ sol[:-4].astype(dtype=int)
            if sum(conds) == 0:
                if buy_day == 0:  # if the buy signal is seen and buy hasn't been initialized
                    trans_counter += 1
                    buy_day = current_day
                    buy_price = t_dataframe.iloc[buy_day + 1]['Open']
                    current_day = int(hold_days + buy_day)
                    sell_day = current_day
                    sell_price = t_dataframe.iloc[sell_day]['Close']
                    profit = (sell_price - buy_price) / buy_price
                    sum_profits += profit
                    buy_day = 0
                    counter += 1
            else:
                current_day += 1
            conds = np.zeros(sol.shape[0] - 1)
        if counter == 0:
            tickers_profits[i] = 0
        else:
            tickers_profits[i] = sum_profits / counter
    fitness = np.mean(tickers_profits)
    return fitness


# PROGRAM STARTS

if not os.path.exists(f"exhaustive_outputs"):
    os.mkdir(f"exhaustive_outputs")

# print("Stock ticker label?")
# ticker = str(input()).upper()
tickers_list = pd.read_csv('tickers.csv')['tickers']
tickers_rules = []
tickers_dataframes = []
for i in range(len(tickers_list)):
    df_ticker = stock_processor(ticker=tickers_list[i])
    tickers_dataframes.append(df_ticker)
    df1 = df_ticker.copy()
    df2 = df_ticker.shift(1)
    df21 = (df1 - df2)

    temp = [
        df21['trend_sma_9'] > 0,
        df21['trend_ema_9'] > 0,
        df21['trend_sma_20'] > 0,
        df21['trend_ema_20'] > 0,
        df21['trend_sma_50'] > 0,
        df21['trend_ema_50'] > 0,
        df21['trend_sma_200'] > 0,
        df21['trend_ema_200'] > 0,
        ((df1['trend_sma_9'] <= df1['trend_ema_9']) & (df2['trend_sma_9'] >= df2['trend_ema_9'])),
        ((df1['trend_ema_9'] <= df1['trend_sma_9']) & (df2['trend_ema_9'] >= df2['trend_sma_9'])),
        ((df1['trend_sma_20'] <= df1['trend_ema_20']) & (df2['trend_sma_20'] >= df2['trend_ema_20'])),
        ((df1['trend_ema_20'] <= df1['trend_sma_20']) & (df2['trend_ema_20'] >= df2['trend_sma_20'])),
        ((df1['trend_sma_50'] <= df1['trend_ema_50']) & (df2['trend_sma_50'] >= df2['trend_ema_50'])),
        ((df1['trend_ema_50'] <= df1['trend_sma_50']) & (df2['trend_ema_50'] >= df2['trend_sma_50'])),
        ((df1['trend_sma_200'] <= df1['trend_ema_200']) & (df2['trend_sma_200'] >= df2['trend_ema_200'])),
        ((df1['trend_ema_200'] <= df1['trend_sma_200']) & (df2['trend_ema_200'] >= df2['trend_sma_200'])),
    ]
    all_rules = np.array(temp).T
    tickers_rules.append(all_rules)

# print("How many rules?")

# temp = [
#     df21['trend_sma_9'] > 0,
#     df21['trend_ema_9'] > 0,
#     df21['trend_sma_20'] > 0,
#     df21['trend_ema_20'] > 0,
#     df21['trend_sma_50'] > 0,
#     df21['trend_ema_50'] > 0,
#     df21['trend_sma_200'] > 0,
#     df21['trend_ema_200'] > 0,
# ]


print("How many days?")
days = int(input())
bin_days = int(np.log2(days))
print(f"binary numbers for days in solutions: {bin_days}")

n_bits = all_rules.shape[1] + bin_days
if not os.path.exists(f"multi_stock_exhaustive_outputs/all_output"):
    os.mkdir(f"multi_stock_exhaustive_outputs/all_output")

n_iter = 2 ** n_bits
# if os.path.exists(f"multi_stock_exhaustive_outputs/all_output/ticker_{ticker}_N{n_bits - bin_days}_Days{days}_scores"):
#     all_scores = np.load(
#         f"multi_stock_exhaustive_outputs/all_output/ticker_{ticker}_N{n_bits - bin_days}_Days{days}_scores.npy")
#     all_bin_sols = np.load(
#         f"multi_stock_exhaustive_outputs/all_output/ticker_{ticker}_N{n_bits - bin_days}_Days{days}_solutions.npy")
# else:
all_scores, all_bin_sols = exhaustive_searcher(stock_fitness, n_bits, n_iter)
# np.save(f"multi_stock_exhaustive_outputs/all_output/ticker_{ticker}_N{n_bits - bin_days}_Days{days}_solutions", all_bin_sols)
# np.save(f"multi_stock_exhaustive_outputs/all_output/ticker_{ticker}_N{n_bits - bin_days}_Days{days}_scores", all_scores)

sorted_scores_idx = np.argsort(all_scores)[::-1]
sorted_scores = all_scores[sorted_scores_idx]
sorted_bin_sols = all_bin_sols[sorted_scores_idx]

binary_format = f'0{n_bits}b'
df_columns = ['Solution #']
for i in range(n_bits):
    df_columns.append(f'R{i + 1}')
df_columns.append('Hold days')
df_columns.append('Average of transactions profit')
df = pd.DataFrame(columns=df_columns)
for i in range(100):
    bin_sol = sorted_bin_sols[i]
    hold_days = 0
    for ele in bin_sol[-4:].astype(dtype=int):
        hold_days = (hold_days << 1) | ele
    hold_days += 1
    new_row_data = np.concatenate(
        [[f'Solution {sorted_scores_idx[i]}'], bin_sol.astype(dtype=int), [hold_days], [f"{sorted_scores[i]:.6f}"]])
    df.loc[len(df.index)] = new_row_data
    # new_row = pd.DataFrame(data=new_row_data, columns=df_columns)
    # print(new_row.head())
    # df = pd.concat([df, new_row], ignore_index=True)
    # df[f"Solution {i + 1}"] = np.concatenate([bin_sol, [hold_days], [f"{sorted_scores[i]:.6f}"]])
df.to_csv(f"multi_stock_exhaustive_outputs/all_output/summary.csv",
          index=False)


def get_solution_dataframe(objective, i):
    sol = np.load(f"./outputs/sol{i}_rules")
    obj_val, df = objective(sol, 0, i)
    # save it to output
    new_row = ['Sum', '', '', '', '', obj_val]
    new_df = pd.DataFrame([new_row], columns=cols)
    df = pd.concat([df, new_df], ignore_index=True)
    df.to_csv(f"./outputs/sol{i}_transactions.csv", index=False)

    return df


def stock_solution_report(sol):
    sol = np.array(sol)
    df = pd.DataFrame(columns=cols)
    tickers_profits = np.zeros(shape=len(tickers_list))
    hold_days = 0
    for ele in sol[-4:].astype(dtype=int):
        hold_days = (hold_days << 1) | ele
    hold_days += 1
    for i in range(len(tickers_list)):
        t_rule = tickers_rules[i]
        t_dataframe = tickers_dataframes[i]
        conds = np.zeros(sol.shape[0] - 1)
        trans_counter = 0
        sum_profits = 0
        counter = 0
        current_day, buy_day, sell_day = 0, 0, 0
        while current_day < t_rule.shape[0]:
            if current_day + hold_days >= t_rule.shape[0]:
                break

            conds = t_rule[current_day].astype(dtype=int) ^ sol[:-4].astype(dtype=int)
            if sum(conds) == 0:
                if buy_day == 0:  # if the buy signal is seen and buy hasn't been initialized
                    trans_counter += 1
                    buy_day = current_day
                    buy_price = t_dataframe.iloc[buy_day + 1]['Open']
                    buy_date = t_dataframe.iloc[buy_day + 1].name.strftime("%m/%d/%Y")
                    current_day = int(hold_days + buy_day)
                    sell_day = current_day
                    sell_price = t_dataframe.iloc[sell_day]['Close']
                    sell_date = t_dataframe.iloc[sell_day].name.strftime("%m/%d/%Y")
                    profit = (sell_price - buy_price) / buy_price
                    sum_profits += profit
                    new_row = [f'{tickers_list[i]}', buy_date, buy_price, sell_date, sell_price, profit]
                    new_df = pd.DataFrame([new_row], columns=cols)
                    df = pd.concat([df, new_df], ignore_index=True)
                    buy_day = 0
                    counter += 1
            else:
                current_day += 1

            conds = np.zeros(sol.shape[0] - 1)

        if counter == 0:
            tickers_profits[i] = 0
        else:
            tickers_profits[i] = sum_profits / counter
    fitness = np.mean(tickers_profits)
    new_row = ['Mean profit', '', '', '', '', fitness]
    new_df = pd.DataFrame([new_row], columns=cols)
    df = pd.concat([df, new_df], ignore_index=True)
    fitness = np.std(tickers_profits)
    new_row = ['Std profit', '', '', '', '', fitness]
    new_df = pd.DataFrame([new_row], columns=cols)
    df = pd.concat([df, new_df], ignore_index=True)
    return df


for i in range(10):
    bin = sorted_bin_sols[i]
    df = stock_solution_report(bin)
    df.to_csv(f"multi_stock_exhaustive_outputs/top5_report/rank{i + 1}.csv", index=False)
