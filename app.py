#!/usr/bin/python
# -*- coding: utf8 -*-
import ast

from flask import Flask, request, jsonify
from model import estimate_price

app = Flask(__name__)


@app.route('/api/estimate_price', methods=['POST'])
def predict():
    if request.method == 'POST':
        input_data = request.form['data']
        data = ast.literal_eval(input_data)
    try:
        price = estimate_price(data)
    except:
        price = None
    return jsonify(price=price)


if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0", port=8001)
