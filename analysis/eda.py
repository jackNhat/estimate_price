from os.path import join, dirname

import pandas as pd


def analyze(columns):
    model_list = columns.to_list()
    brands = list(set([i.split(" ")[0] for i in model_list]))
    models = sorted(list(set(["-".join(i for i in i.split(" ")[:3])
                              for i in model_list])))
    return brands, models


colnames = ['brands', 'model', 'chip_brands', 'chip_speed', 'chip_type', 'ram', 'hdd', 'ssd', 'card_brand', 'card_model',	'monitorSize', 'price']
file_name = join(dirname(dirname(__file__)), "data", "laptop_data.xlsx")
data = pd.read_excel(file_name)
# model_brands, models_type = analyze(columns=data["model"])
# chip_brands, chip_type = analyze(columns=data["chip"])
card_brands, card_type = analyze(columns=data["card"])
print(0)
