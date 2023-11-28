import pandas as pd
import os

temp = pd.read_parquet('/data/other_data/bar_data_56_tickers.parquet.gzip')
all_ticker = temp.index.levels[0]
cons = 780*13

for j in range(300):
    output = pd.DataFrame({all_ticker[0]:temp.xs(key=all_ticker[0], level=0).iloc[(j*780):(cons+j*780)]['close']})
    idx = 1

    for i in range(len(all_ticker)-1):
        temp_AAL = temp.xs(key=all_ticker[idx], level=0)
        temp_AAL = temp_AAL.iloc[(j*780):(cons+j*780)]['close']
        output[all_ticker[idx]] = temp_AAL
        idx += 1

    output = output.ffill()
    output_filled = output.apply(lambda x: x.fillna(x.dropna().iloc[0]) if x.first_valid_index() is not None else x)
    output_filled.to_csv('dataset/test_56_ticker_13_days.csv', header=None, index=False)
    print('====== Start to train and predict ======')

    os.system('python main.py --train True --evaluate True --dataset test_56_ticker_13_days  --window_size 144 --horizon 36 --norm_method z_score --train_length 7 --valid_length 2 --test_length 1 --device cuda')

    print('====== Train finished and start to save file ======')
    test_t = pd.read_csv('output/test_56_ticker_13_days/test/target.csv', header=None)
    test_p = pd.read_csv('output/test_56_ticker_13_days/test/predict.csv', header=None)

    def is_similar(row1, row2, threshold=0.0001):
        """Check if two rows are similar within a percentage threshold."""
        for a, b in zip(row1, row2):
            try:
                if not (abs(a - b) / max(abs(a), abs(b)) <= threshold):
                    return False
            except TypeError:
                # For non-numeric types, check for exact match
                if a != b:
                    return False
        return True

    # for index, row in output_filled.iterrows():
    #     if is_similar(row, train_t.iloc[0]):
    #         print(index)

    for idxx, row in output_filled.iterrows():
        if is_similar(row, test_t.iloc[0]):
            test_p = test_p.set_index(output_filled.loc[idxx:].index)
            test_p.columns = all_ticker
            test_p.to_parquet('jiahong/prediction_{}.parquet'.format(j))
            break