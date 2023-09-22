import joblib 
import numpy as np
import pandas as pd
from pathlib import Path

#loading model and doing prediction after taking the data from user.

class PredictionPipeline:
    def __init__(self):
        self.model = joblib.load(Path('artifacts/model_trainer/model.joblib'))

    
    def predict(self, data):
        prediction = self.model.predict(data)

        return prediction