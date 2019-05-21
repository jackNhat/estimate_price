from os.path import join, dirname

import joblib
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import mean_squared_error


def convert_data(df):
    def to_integer(df_columns):
        return pd.to_numeric(df_columns, errors='coerce', downcast='integer')

    def to_float(df_columns):
        return pd.to_numeric(df_columns, errors='coerce', downcast='float')

    df.ram = to_integer(df.ram)
    df.hdd = to_integer(df.hdd)
    df.ssd = to_integer(df.ssd)
    df.monitorSize = to_float(df.monitorSize)
    df.price = to_float(df.price)
    df.chip_speed = to_float(df.chip_speed)
    return df


if __name__ == '__main__':
    train_path = join(dirname(__file__), "data", "laptop_data.xlsx")
    col_names = ['brands', 'product_name', 'chip_brands', 'chip_speed', 'chip_type', 'ram', 'hdd', 'ssd', 'card_brand',
                 'card_model', 'monitorSize', 'price']
    content = pd.read_excel(train_path, names=col_names)

    data = convert_data(df=content)
    X = data[['brands', 'product_name', 'chip_brands', 'chip_speed', 'chip_type', 'ram', 'hdd', 'ssd', 'card_brand',
              'card_model', 'monitorSize']]
    y = data.price
    X = pd.get_dummies(data=X)
    print("Shape of X: ", X.shape)
    transformer_path = join(dirname(__file__), "transformer.pkl")
    joblib.dump(X, transformer_path)

    X_train, X_dev, y_train, y_dev = train_test_split(X, y, test_size=.20, random_state=42)

    # Gradient Boosting Regression
    model = GradientBoostingRegressor(loss='ls', max_depth=6)
    model.fit(X_train, y_train)
    gbr_model = join(dirname(__file__), "gbr_model.pkl")

    joblib.dump(model, gbr_model)

    predicted = model.predict(X_dev)
    residual = y_dev - predicted

    rmse = np.sqrt(mean_squared_error(y_dev, predicted))
    scores = cross_val_score(model, X_dev, y_dev, cv=12)

    print('\nCross Validation Scores:')
    print(scores)
    print('\nMean Score:')
    print(scores.mean())
    print('\nRMSE:')
    print(rmse)
