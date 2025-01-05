import os
import json
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import joblib

class Main(object):
    def __init__(self):
        self.cur_dir = os.path.dirname(os.path.realpath(__file__))
        self.model = joblib.load(os.path.join(self.cur_dir, './model/linear_regression_model.pkl'))
    
    def predict(self, mlskill_input):
        data = pd.read_json(mlskill_input)
        predictions = self.model.predict(data.values)
        return json.dumps(predictions.tolist())
