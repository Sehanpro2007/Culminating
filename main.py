import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from numpy.random import sample
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler

ds = pd.read_csv('venv/parkinsons.data')
print(ds)

# create multiple charts in order to try coming up with model for the program to run
# for label in ds.columns:
# plt.plot(ds[ds['status'] == 1][label], color='blue', label='parkinson', alpha=0.7)
# plt.plot(ds[ds['status'] == 0][label], color='red', label='no parkinson', alpha=0.7)
# plt.title(label)
# plt.ylabel("Probability")
# plt.xlabel(label)
# plt.legend()
# plt.show()

# split my dataframe into 3 separate testing data
train, validate, test = np.split(ds.sample(frac=1), (int(0.6 * len(ds)), int(0.8 * len(ds))))


# scale all numbers in order to keep all datapoint relative to the average

# create scaling function
def scale(dataframe):
    ds_x = dataframe.drop("status")
    fx = dataframe[ds_x].values
    y = dataframe[ds['status']].values
    scaler = StandardScaler()
    fx = scaler.fit_transform(fx)
    data = np.hstack((fx, np.reshape(y, (-1, 1))))
    return fx, y, data


scale(ds)
