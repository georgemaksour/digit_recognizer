import pandas as pd

from config import *
from utils import *

logging.basicConfig(filename='output/log.txt', filemode='w')

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    train_set = pd.read_csv('data/train.csv')
    test_set = pd.read_csv('data/test.csv')

    train_target = train_set.pop('label')
    train_target = to_categorical(train_target)

    train_ds = train_set.values
    test_ds = test_set.values

    train_ds = train_ds.reshape((train_ds.shape[0], 28, 28, 1))
    test_ds = test_ds.reshape((test_ds.shape[0], 28, 28, 1))

    train_ds, test_ds = prep_pixels(train_ds, test_ds)

    scores, histories, preds = evaluate_model(train_ds, train_target, test_ds)
    preds_df = pd.DataFrame(preds)
    preds = []
    for (index, row), (j, row_two) in zip(preds_df.iterrows(), train_set.iterrows()):
        numbs = list(row[1])
        pred_num = numbs.index(max(numbs))
        preds.append(pred_num)
    preds = enumerate(preds, 1)
    preds_df = pd.DataFrame(preds, columns=['ImageId', 'Label'])
    preds_df.to_csv('output/preds.csv', index=False)

    summarize_diagnostics(histories)
    summarize_performance(scores)

