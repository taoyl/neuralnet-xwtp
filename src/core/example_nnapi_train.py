#
# Example code for using API to train a new neural network
#

import matplotlib.pyplot as plt
from datarw import DataHandler
from nnapi import NnApi

def show_predicted_data(actual, pred):
    print("Predict and show the result...")
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(actual, 'g--')  # Actual value
    ax.plot(pred, 'r^')  # Predicted value
    ax.set_title("Prediction Result")
    plt.legend((r'Actual', r'Predicted'))
    plt.show()

# Step1: construct your training data
# The data format is like the following:
# [ {'input': [d1, d2, d3, d4...], 'output' : df},
#   {'input': [d1, d2, d3, d4...], 'output' : df},
#   ...
# ]
dataloader = DataHandler("../../data/data.csv")
training_data, test_data = dataloader.load()
tr_data = [{'input': d[:-1], 'output': d[-1]} for d in training_data]
pred_data = [{'input': d[:-1], 'output': d[-1]} for d in test_data]

# Step2. Initialize neural network
api = NnApi(tr_data)
print(api.training_x.shape, api.training_y.shape)

# Step3: train your network
# tol0 is the expected error between predicted output and actual output
# returned value is the final residual between predicted output and actual output
status, residual = api.train(tol0=0.1)
if status:
    print("Final residual is", residual)
else:
    print("Training failed")
    exit(-1)

# Step4: (Option) save your trained network
api.save_network('../../dump.nf')
print("Saving network done")

# Step5: predict your data
# pred_data has the same format as tr_data, you can put None to output
pred_y = api.predict(pred_data)
show_predicted_data([d['output'] for d in pred_data], pred_y)


