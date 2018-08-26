# Use tensorflow DNNRegressor to implament the model

import argparse
import matplotlib.pyplot as plt
import tensorflow as tf

import wtp_data

parser = argparse.ArgumentParser()
parser.add_argument('--file', default='../../data/data.csv', type=str, help='data file')
parser.add_argument('--batch_size', default=100, type=int, help='batch size')
parser.add_argument('--train_steps', default=10000, type=int, help='number of training steps')
parser.add_argument('--predict', action='store_true', default=False, help='Predict and show result')


def show_predictions(predictions, expected=None):
    """Show prediction result v.s expected values"""

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_title("Prediction Result")

    pred_val = [p['predictions'][0] for p in predictions]
    ax.plot(pred_val, 'r^')  # Predicted value
    if expected is not None:
        ax.plot(list(expected), 'g--')  # Actual value
        plt.legend((r'Predicted', r'Expetced'))
    else:
        plt.legend(r'Predictions')

    plt.show()


def main(argv):
    """main"""

    args = parser.parse_args(argv[1:])
    (x_train, y_train), (x_test, y_test) = wtp_data.load_data(fpath=args.file)

    train_input_fn = wtp_data.make_dataset(args.batch_size, x=x_train, y=y_train, shuffle=True)
    test_input_fn = wtp_data.make_dataset(args.batch_size, x=x_test, y=y_test)

    # define fature columns, excluding the last column
    feature_columns = [tf.feature_column.numeric_column(key=item[0])
            for item in list(wtp_data.DATA_COLUMNS.items())[:-1]]
    print(feature_columns)

    model = tf.estimator.DNNRegressor(
        feature_columns=feature_columns,
        hidden_units=[10, 10],
        model_dir='./model'
    )

    # train
    model.train(input_fn=train_input_fn, steps=args.train_steps)

    # evaluate
    eval_result = model.evaluate(input_fn=test_input_fn)
    print(eval_result)

    # predict
    if args.predict:
        predict_input_fn = wtp_data.make_dataset(args.batch_size, x=x_test)
        predictions = model.predict(input_fn=predict_input_fn)
        show_predictions(predictions, y_test)

if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run(main=main)
