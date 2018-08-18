from neuralnetwork import ShapeError, NeuralNetwork
from datarw import DataHandler
import matplotlib.pyplot as plt
import numpy as np
import os
import argparse
import sys

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--file', dest='data_file', help='data file (*.csv)', type=str)
    # parser.add_argument('-h', '--help', help='usage help', action='store_true')
    args = parser.parse_args()
    # print(args.data_file)

    if not os.path.exists(args.data_file):
        sys.exit(1)

    dataloader = DataHandler(args.data_file)
    training_data, test_data = dataloader.load()
    # print(training_data)
    training_x, training_y = DataHandler.split_xy(training_data, True)
    test_x, test_y = DataHandler.split_xy(test_data, True)
    print(training_x.shape, training_y.shape)
    print(test_x.shape, test_y.shape)

    net = [(3, ''), (4, 'relu'), (4, 'relu'), (1, 'feedthrough')]
    print("Initializing neural network...")
    try:
        network = NeuralNetwork(training_x, training_y, sizes=net)
    except ShapeError as e:
        print(e)
        sys.exit(99)

    print("Training the neural network...")
    retry = 3
    while retry:
        retry -= 1
        residual, mu, epoch, msg = network.train(epoch=2000, retry=5, mu0=0.1, beta=10, tol=0.1)
        if residual is None:
            print("Training failed: {}".format(msg))
            if retry == 0:
                sys.exit(101)
            else:
                network.randomize_wb()
                continue
        elif residual > 0.1:
            print("\nTraining failed: optimization misconvergence!")
            if retry == 0:
                sys.exit(102)
            else:
                network.randomize_wb()
                continue
        else:
            print("Training succeed, residual error={}, epoch={}".format(residual, epoch))
            break

    #network.dump("dump.nf")
    # network.load("dump")
    print("Predict and show the result...")
    z1 = network.predict(test_x)


    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(test_y[0], 'g--')  # 真值
    ax.plot(z1[0], 'r^')  # 预测值
    ax.set_title("Prediction Result")
    plt.legend((r'Actual', r'Predicted'))
    # err_percent = (z1 - test_y) * 100 / test_y
    # ax.errorbar(np.arange(len(z1[0])), np.zeros(len(z1[0])), yerr=err_percent[0])
    plt.show()

if __name__ == '__main__':
    main()
