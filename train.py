from os.path import join, dirname

import joblib
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import mean_squared_error


def convert_data(df):
    def to_numberic(df_columns):
        return pd.to_numeric(df_columns, errors='coerce', downcast='integer')

    def to_float(df_columns):
        return pd.to_numeric(df_columns, errors='coerce', downcast='float')

    df.monitorSize = to_numberic(df.monitorSize)
    df.ram = to_numberic(df.ram)
    df.hdd = to_numberic(df.hdd)
    df.ssd = to_numberic(df.ssd)
    df.price = to_numberic(df.price)
    df.chip_speed = to_float(df.chip_speed)
    return df


if __name__ == '__main__':
    train_path = join(dirname(__file__), "data", "train.xlsx")
    col_names = ['brands', 'model', 'chip_brands', 'chip_speed', 'chip_type', 'ram', 'hdd', 'ssd', 'card_brand',
                 'card_model', 'monitorSize', 'price']
    content = pd.read_excel(train_path, names=col_names.append('price'))

    data = convert_data(df=content)
    X = data[['brands', 'chip_brands', 'chip_speed', 'ram', 'hdd', 'ssd', 'card_brand', 'card_model', 'monitorSize']]
    y = data.price
    X = pd.get_dummies(data=X)
    print("Shape of X: ", X.shape)
    transformer_path = join(dirname(__file__), "model", "transformer.pkl")
    joblib.dump(X, transformer_path)

    X_train, X_dev, y_train, y_dev = train_test_split(X, y, test_size=.20, random_state=42)
    gbr = GradientBoostingRegressor(loss='ls', max_depth=6)
    gbr.fit(X_train, y_train)

    model_path = join(dirname(__file__), "model", "gbr_model.pkl")
    joblib.dump(gbr, model_path)

    predicted = gbr.predict(X_dev)
    residual = y_dev - predicted

    rmse = np.sqrt(mean_squared_error(y_dev, predicted))
    scores = cross_val_score(gbr, X_dev, y_dev, cv=12)

    print('\nCross Validation Scores:')
    print(scores)
    print('\nMean Score:')
    print(scores.mean())
    print('\nRMSE:')
    print(rmse)
