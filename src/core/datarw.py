# Load data
import re
import random
# import csv
import numpy as np


class DataHandler:
    def __init__(self, file):
        self.file = file
        # print(file)

    def load(self, trdata_percent=70, shuffle=False, cb=None):
        pattern = re.compile(r'^\s*(-?[\d.]+)')
        line_num, invalid_num = (0, 0)
        data = []
        with open(self.file, 'r') as fh:
            for line in fh:
                line_num += 1
                items = line.strip('\n').split(r',')
                # validation check
                valid = True
                for d in items:
                    if not pattern.match(d):
                        valid = False
                        invalid_num += 1
                        break
                if valid:
                    data.append([float(x) for x in items])
                    #print("LINE{}: {}".format(line_num, items))
        if cb:
            cb("Data parsing: {0}/{1} invalid lines found"
                .format(invalid_num, line_num), 'blue', 'I')
        else:
            print("Data parsing: {0}/{1} invalid lines found".format(invalid_num, line_num))
        if shuffle:
            random.shuffle(data)
        tr_size = int(len(data) * trdata_percent / 100)
        tr_set = np.array(data[:tr_size])
        # print(tr_set)
        test_set = np.array(data[tr_size:])
        return tr_set, test_set

    @staticmethod
    def split_xy(dataset, shuffle=False):
        """Split the dataset into x and y. The returned data is N x M,
        where M is the total number of data"""
        ds = dataset
        if shuffle:
            np.random.shuffle(ds)
        x = ds[:, :-1].transpose()
        y = ds[:, -1].reshape(1, ds.shape[0])
        return x, y

    @staticmethod
    def save(data, fname):
        rows = data.transpose()
        with open(fname, 'w') as fh:
            #writer = csv.writer(fh, delimiter=',')
            # write header
            # writer.writerow(["hour", "outdoor temp", "avg temp", "predict"])
            #writer.writerows(rows)
            for row in rows:
                fh.write(','.join([str(x) for x in row]))
                fh.write("\n")
        return True
