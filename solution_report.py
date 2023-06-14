import numpy as np
import pandas as pd

from stock_processor import stock_processor

# np.load()


cols = ['T', 'Buy date', 'Buy price', 'Sell date', 'Sell price', 'Profit Ratio']
tickers_list = pd.read_csv('tickers.csv')['tickers']
tickers_conds = []
for i in range(len(tickers_list)):
    ticker_name = tickers_list[i]
    df_ticker = stock_processor(ticker=ticker_name)
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
    tickers_conds.append(all_rules)

top_sols = np.array([139278, 139279, 139277, 878607, 876558, 876559 , 139535, 141326 , 335884, 141323])

print("How many days?")
days = int(input())
bin_days = int(np.log2(days))
print(f"binary numbers for days in solutions: {bin_days}")

n_bits = all_rules.shape[1] + bin_days

n_iter = 2 ** n_bits

binary_format = f'0{n_bits}b'
sorted_bin_sols = np.zeros((n_iter, n_bits))
for i in range(5):
    sol = np.array(list(format(top_sols[i], binary_format))).astype(int)
    sorted_bin_sols[i] = sol


def stock_solution_report(sol):
    sol = np.array(sol)

    df = pd.DataFrame(columns=cols)

    tickers_profits = np.zeros(shape=len(tickers_list))
    current_day, buy_day, sell_day = 0, 0, 0
    conds = np.zeros(sol.shape[0] - 1)
    trans_counter = 0
    hold_days = 0
    for ele in sol[-4:].astype(dtype=int):
        hold_days = (hold_days << 1) | ele

    # print(hold_days, sol[-4:])
    hold_days += 1

    for i in range(len(tickers_list)):
        ticker_name = tickers_list[i]
        ticker_rules = tickers_conds[i]
        sum_profits = 0
        counter = 0
        while current_day < df1.shape[0]:
            # counter += 1
            if current_day + hold_days >= df1.shape[0]:
                break

            conds = ticker_rules[current_day].astype(dtype=int) ^ sol[:-4].astype(dtype=int)
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
                    # all_profit.append(profit)
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

    new_row = ['Average profit', '', '', '', '', np.mean(tickers_profits)]
    new_df = pd.DataFrame([new_row], columns=cols)
    df = pd.concat([df, new_df], ignore_index=True)
    return df


for i in range(10):
    bin = sorted_bin_sols[i]
    df = stock_solution_report(bin)
    df.to_csv(f"multi_stock_exhaustive_outputs/top5_report/rank{i + 1}.csv", index=False)
