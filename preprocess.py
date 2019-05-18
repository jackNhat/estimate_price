from os.path import join, dirname
import pandas as pd
from sklearn.model_selection import train_test_split

file_name = join(dirname(__file__), "data", "laptop_data.xlsx")
data = pd.read_excel(file_name)
train, test = train_test_split(data, train_size=0.8, shuffle=True)
train_file = join(dirname(__file__), "data", "train.xlsx")
test_file = join(dirname(__file__), "data", "test.xlsx")
train.to_excel(train_file, index=False)
test.to_excel(test_file, index=False)
