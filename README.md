***Consumer Classifier*** is Python module for performing clustering (grouping) of consumers based on their behaviour features.


## Installation
***

### Dependencies

Package requires:

* Python (>= 3.7)
* sklearn (= 0.0)
* Pandas (>= 1.0.5)
* Matplotlib (>= 3.3.2)

## Usuage

To run clustering followed by classification 

`python cluster_classify.py --dataset_path "data/Technical test sample data.csv" --output_dir test_out1 --ignore_features "consumer_id,gender,account_status,customer_age" `

To run model inferencing with new data

`python predict.py --output_dir pred_out --model_path model_store/model.pkl --data_path data/test.csv `
