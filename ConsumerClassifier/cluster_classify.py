import pandas as pd
from pickle import dump
from pathlib import Path
import argparse
import json

from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
from sklearn.ensemble import RandomForestClassifier

class Modelling:
    def __init__(self, dataset_path, output_dir):
        self.dataset_path = dataset_path

        if Path(self.dataset_path).suffix == '.ods':
            df_ = pd.read_excel(self.dataset_path, engine="odf")
        elif Path(self.dataset_path).suffix == '.csv':
            df_ = pd.read_csv(self.dataset_path)
        else:
            raise Exception('Not a valid file type - ', Path(self.data_path).suffix)

        self.data = df_
        self.output_dir = output_dir
        self.model_log = {}
        
        if not Path(self.output_dir).exists():
            Path(self.output_dir).mkdir()
        
    def prep_data_train(self):

        X = self.data[self.selected_columns].values
        y = self.data.cluster.values
        return X, y

    def kmeans_clustering(self, ignore_cols, clusters_n=2):
        
        self.selected_columns = self.data.columns.difference(ignore_cols)
        self.model_log["feature_colums"] = tuple(self.selected_columns)
        self.model_log["model_hyperparameteres"]  = {"cluster_size": clusters_n}

        train_df = self.data[self.selected_columns]

        kmeans = KMeans(n_clusters=clusters_n, init="random",n_init=10,max_iter=10000, random_state=5)
        clusters = kmeans.fit(train_df)
        self.data['cluster'] = kmeans.predict(train_df)


    def random_forest_classification(self, estimators=10, test_size=0.25):

        X, y = self.prep_data_train()

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)

        model = RandomForestClassifier(random_state=5, n_estimators=estimators)
        model.fit(X_train, y_train)
        
        self.model_log["model_hyperparameteres"].update({"rf_estimators": estimators})
        self.metric_calculation(model, {"x test": X_test, "y test": y_test})

        return model

    def metric_calculation(self, model, test_data):
        predicted = model.predict(test_data["x test"])
        accuracy = accuracy_score(test_data["y test"], predicted)
        f1_score_ = f1_score(test_data["y test"], predicted, average='macro')
        recall = recall_score(test_data["y test"], predicted, average="binary")
        precision = precision_score(test_data["y test"], predicted, average="binary")
        
        print("Accuracy Score - ", accuracy)
        print("f1_score - ", f1_score_)
        print("Recall - ", recall)
        print("Precision - ", precision)

        self.model_log["metrics"] = {"accuracy": accuracy, "f1_score": f1_score_, "Recall": recall, "Precision": precision}

    def save_model(self, model, model_name="model"):

        dump(model, open(Path(self.output_dir, model_name).with_suffix('.pkl'), 'wb'))

        cluster_ = self.data['cluster'].value_counts().to_dict()
        for key in cluster_.keys():
            cluster_[key] = "cluster_" + str(cluster_[key])

        self.model_log["classes"] =  cluster_

        with open(Path(self.output_dir, 'model.json'), 'w') as outfile:
            json.dump(self.model_log, outfile, indent=4)



if __name__ =='__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dataset_path', type=str)
    parser.add_argument('-o', '--output_dir', type=str)
    parser.add_argument('-ig', '--ignore_features', help='delimited list input', type=lambda s: [str(item) for item in s.split(',')])
    args = parser.parse_args()
    
    output_dir = args.output_dir
    dataset_path = args.dataset_path
    ignore_features = args.ignore_features

    modelling = Modelling(dataset_path, output_dir)
    modelling.kmeans_clustering(ignore_cols=ignore_features, clusters_n=2)
    model = modelling.random_forest_classification()
    modelling.save_model(model)
