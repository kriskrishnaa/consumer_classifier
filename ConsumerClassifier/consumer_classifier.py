from pathlib import Path
import os
from pickle import load
import json
import pandas as pd

class classifier:
    def __init__(self, model_path, data_path, output_dir):
        self.model_path = model_path
        self.data_path = data_path
        self.output_dir = output_dir
        if not Path(self.output_dir).exists():
            Path(self.output_dir).mkdir()
    
    def load_model(self, model_log_path="", **kwargs):
        
        if not model_log_path:
            model_log_path=Path(self.model_path).with_suffix('.json')
            
        model = load(open(self.model_path, 'rb'))
        with open(model_log_path) as json_file:
            model_log = json.load(json_file) 
        return model, model_log
    
    def prep_data(self, model_log):
        expected_columns = list(model_log['feature_colums'])
        
        if Path(self.data_path).suffix == '.ods':
            df_ = pd.read_excel(self.data_path, engine="odf")
        elif Path(self.data_path).suffix == '.csv':
            df_ = pd.read_csv(self.data_path)
        else:
            raise Exception('Not a valid file type - ', Path(self.data_path).suffix)
            
        df_ = df_[expected_columns]
        return df_.values

    def predict(self, model_path='model.pkl', **kwargs):
        model, model_log = self.load_model()
        predictData = self.prep_data(model_log)
        pred_arr = model.predict(predictData)
        pred = [model_log["classes"][str(i)] for i in pred_arr]

        with open(Path(self.output_dir, "output.txt"), 'w') as f:
            for item in pred:
                f.write("%s\n" % item)
