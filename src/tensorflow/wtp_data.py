# Data model
# nerotao@foxmail.com

from collections import OrderedDict
import numpy as np
import pandas as pd
import tensorflow as tf

DATA_COLUMNS = OrderedDict([
    ('hour', np.int32),
    ('room_temp', np.float32),
    ('avg_temp', np.float32),
    ('set_temp', np.float32),
])

def load_data(fpath, y='set_temp', train_percent=70, seed=None):
    """Load WTP data"""

    df = pd.read_csv(fpath, names=DATA_COLUMNS.keys(), dtype=DATA_COLUMNS,header=1)
    # drop N/A data
    data = df.dropna()

    # shuffle seed
    np.random.seed(seed)

    # generare train and test data
    x_train = data.sample(frac=train_percent/100.0, random_state=seed)
    x_test = data.drop(x_train.index)
    y_train = x_train.pop(y)
    y_test = x_test.pop(y)
    print("train set: {}-{}, test set: {}-{}".format(x_train.shape, y_train.shape, x_test.shape, y_test.shape))

    return (x_train, y_train), (x_test, y_test)


def make_dataset(batch_size, x, y=None, shuffle=False, shuffle_buffer_size=1000):
    """Create a dataset"""
    
    def input_fn():
        if y is not None:
            ds = tf.data.Dataset.from_tensor_slices((dict(x), y))
        else:
            ds = tf.data.Dataset.from_tensor_slices(dict(x))

        if shuffle:
            ds = ds.shuffle(shuffle_buffer_size).batch(batch_size).repeat()
        else:
            ds = ds.batch(batch_size)

        return ds.make_one_shot_iterator().get_next()

    return input_fn





    
