#
# Example code for using API to load a trained network
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

# Step3: load trained network
status = api.load_network("../../dump.nf")
if status:
    print("Loading network done")
else:
    print("Loading network failed")
    exit(-1)

# Step4: predict your data
# pred_data has the same format as tr_data, you can put None to output
pred_y = api.predict(pred_data)
show_predicted_data([d['output'] for d in pred_data], pred_y)

