from os.path import join, dirname

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from gbr_model import estimate_price


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


test_path = join(dirname(dirname(__file__)), "data", "test.xlsx")
col_names = ['brands', 'product_name', 'chip_brands', 'chip_speed', 'chip_type', 'ram', 'hdd', 'ssd', 'card_brand',
             'card_model', 'monitorSize', 'price']

content = pd.read_excel(test_path, names=col_names)
data = convert_data(df=content)
X_cols = ['brands', 'product_name', 'chip_brands', 'chip_speed', 'chip_type', 'ram', 'hdd', 'ssd', 'card_brand',
          'card_model', 'monitorSize']
X_test = data[X_cols]
y_test = data.price.values.reshape(-1, 1)
X_test = X_test.values.tolist()
y_pred = []
for x in X_test:
    x_test = dict(zip(X_cols, x))
    x_test['card_model'] = str(x_test['card_model'])
    pred = estimate_price(x_test)
    y_pred.append(pred)
y_pred = np.array(y_pred).reshape(-1, 1)
plt.rcParams['figure.figsize'] = 16, 5
plt.figure()
plt.plot(y_test[400:500], label="Real")
plt.plot(y_pred[400:500], label="Predicted")
plt.legend()
plt.title("Price: real vs predicted")
plt.ylabel("Price")
plt.xticks(())
plt.savefig("evaluate.png")
