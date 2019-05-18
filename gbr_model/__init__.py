from os.path import join, dirname

import joblib
import numpy as np

model_path = join(dirname(__file__), "gbr_model.pkl")
transformer_path = join(dirname(__file__), "transformer.pkl")
model = joblib.load(model_path)
transformer = joblib.load(transformer_path)


def input_to_one_hot(data):
    # initialize the target vector with zero values
    enc_input = np.zeros(1085)
    enc_input[0] = data['chip_speed']
    enc_input[1] = data['ram']
    enc_input[2] = data['hdd']
    enc_input[3] = data['ssd']
    enc_input[4] = data['monitorSize']

    redefinded_user_input = 'brands_' + data['brands']
    brands_column_index = transformer.columns.tolist().index(redefinded_user_input)
    enc_input[brands_column_index] = 1

    redefinded_user_input = 'chip_type_' + data['chip_type']
    chip_type_column_index = transformer.columns.tolist().index(redefinded_user_input)
    enc_input[chip_type_column_index] = 1

    redefinded_user_input = 'product_name_' + data['product_name']
    model_column_index = transformer.columns.tolist().index(redefinded_user_input)
    enc_input[model_column_index] = 1

    redefinded_user_input = 'chip_brands_' + data['chip_brands']
    chip_brands_column_index = transformer.columns.tolist().index(redefinded_user_input)
    enc_input[chip_brands_column_index] = 1

    redefinded_user_input = 'card_brand_' + data['card_brand']
    card_brand_column_index = transformer.columns.tolist().index(redefinded_user_input)
    enc_input[card_brand_column_index] = 1

    redefinded_user_input = 'card_model_' + data['card_model']
    card_model_column_index = transformer.columns.tolist().index(redefinded_user_input)
    enc_input[card_model_column_index] = 1
    return enc_input


def estimate_price(user_input):
    """
    Estimate price laptop
    :param user_input: dictionary
    Laptop Requirements: chip, ram , monitor size, etc...
    :return: price value
    """
    input_encode = input_to_one_hot(user_input)
    price_pred = model.predict([input_encode])[0]
    return round(price_pred, 2)
