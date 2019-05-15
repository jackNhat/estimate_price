import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
import tensorflowjs as tfjs
import argparse
import warnings

from helpers import final_data

warnings.filterwarnings("ignore")

ap = argparse.ArgumentParser()
ap.add_argument("-p", "--path", required=True,
                help="path to data file")
ap.add_argument("-hp", "--h5path", required=True,
                help="path where keras model is to be saved")
ap.add_argument("-tfp", "--tfjspath", required=True,
                help="path where tfjs layer is to be saved")
args = vars(ap.parse_args())


def data_M1(data):
    data_ = data
    # Method One
    X_train = data_.mean(axis=1)
    X_train = np.reshape(X_train.values, (len(X_train), 1))
    scaler = MinMaxScaler(feature_range=(0, 1))
    X = scaler.fit_transform(X_train)

    # As our focus is on price
    y_train = np.reshape(data['price'].values, (len(X_train), 1))
    scaler = MinMaxScaler(feature_range=(0, 1))
    y = scaler.fit_transform(y_train)

    return X, y


# convert an array of values into a dataset matrix
def create_dataset(dataset, look_back=1):
    X, y = [], []
    for i in range(len(dataset) - look_back - 1):
        a = dataset[i:(i + look_back), 0]
        X.append(a)
        y.append(dataset[i + look_back, 0])
    return np.array(X), np.array(y)


# following split train test function can be used when you use create_dataset_new function
def train_test_split(data_x, data_y, train_size=0.8):
    train_size = int(len(data_x) * train_size)
    X_train, X_test = data_x[0:train_size, :], data_x[train_size:len(data_x), :]
    print(len(X_train), len(X_test))

    y_train, y_test = data_y[0:train_size, :], data_y[train_size:len(data_y), :]
    print(len(y_train), len(y_test))

    return X_train, y_train, X_test, y_test


def run_model(data, h5_path, tfjs_path):
    # use saleprice only for both x and y features ###METHOD 2
    data_ = data
    # As our focus is on Sale Price
    y_train = np.reshape(data_['price'].values, (len(data_), 1))
    scaler = MinMaxScaler(feature_range=(0, 1))
    y = scaler.fit_transform(y_train)
    # split data
    train_size = int(len(y) * 0.8)
    # test_size = len(data_y) - train_size
    train, test = y[0:train_size, :], y[train_size:len(y), :]
    print(len(train), len(test))
    # prepare dataset for LSTM model
    look_back = 25
    X_train, y_train = create_dataset(train, look_back=look_back)
    X_test, y_test = create_dataset(test, look_back=look_back)

    # data_X, data_Y = data_M1(data)
    # train_X, train_y, test_X, test_y = train_test_split(data_X, data_Y)
    # print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)
    # look_back = 20
    # trainX, trainY = create_dataset(train_X, train_y, look_back=look_back)
    # testX, testY = create_dataset(test_X, test_y, look_back=look_back)

    # reshape input to be [samples, time steps, features]
    X_train = np.reshape(X_train, (X_train.shape[0], 1, X_train.shape[1]))
    X_test = np.reshape(X_test, (X_test.shape[0], 1, X_test.shape[1]))

    # LSTM MODEL
    model = Sequential()
    model.add(LSTM(288, input_shape=(1, look_back), return_sequences=True))
    model.add(LSTM(288))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    model.fit(X_train, y_train, epochs=300, batch_size=256, verbose=2)

    # save model to h5py to make restAPI
    model.save(h5_path)  # path where to store h5 model

    # convert model to tfjs layer format
    tfjs_target_dir = tfjs_path
    tfjs.converters.save_keras_model(model, tfjs_target_dir)  # path where to store tfjs format

    # prediction
    train_predict = model.predict(X_train)
    test_predict = model.predict(X_test)
    # inverse tranformation for coreect prediction
    train_predict = scaler.inverse_transform(train_predict)
    y_train = scaler.inverse_transform([y_train])
    test_predict = scaler.inverse_transform(test_predict)
    y_test = scaler.inverse_transform([y_test])

    train_score = math.sqrt(mean_squared_error(y_train[0], train_predict[:, 0]))
    print('Train Score: %.2f RMSE' % train_score)
    test_score = math.sqrt(mean_squared_error(y_test[0], test_predict[:, 0]))
    print('Test Score: %.2f RMSE' % test_score)

    # shift train predictions for plotting
    train_predict_plot = np.empty_like(y)
    train_predict_plot[:, :] = np.nan
    train_predict_plot[look_back:len(train_predict) + look_back, :] = train_predict

    # shift test predictions for plotting
    test_predict_plot = np.empty_like(y)
    test_predict_plot[:, :] = np.nan
    test_predict_plot[len(train_predict) + (look_back * 2) + 1:len(y) - 1, :] = test_predict

    # plot training actual and predicted part
    plt.plot(data['price'], label='Actual')
    plt.plot(pd.DataFrame(train_predict_plot, columns=["close"], index=data.index).close, label='Training')
    plt.plot(pd.DataFrame(test_predict_plot, columns=["close"], index=data.index).close, label='Testing')
    plt.legend(loc='best')
    plt.show()


if __name__ == '__main__':
    # load preprocessed data
    data = final_data(args["path"])

    run_model(data, args["h5path"], args["tfjspath"])
