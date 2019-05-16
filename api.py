#!/usr/bin/python
# -*- coding: utf8 -*-
import json

from flask import Flask, request, jsonify
from model import estimate_price

app = Flask(__name__)


@app.route('/api/estimate_price', methods=['GET'])
def predict():
    if request.method == 'GET':
        input_data = request.args.get('user_input')
        data = json.load(input_data)
    try:
        price = estimate_price(data)
    except:
        price = None
    return jsonify(price=price)


if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0", port=8001)
