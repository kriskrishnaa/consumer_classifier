***Consumer Classifier*** is Python module for performing clustering (grouping) of consumers based on their behaviour features.


## Installation
***

### Dependencies

Package requires:

* Python (>= 3.7)
* sklearn (= 0.0)
* Pandas (>= 1.0.5)
* Matplotlib (>= 3.3.2)
* odfpy (>= 1.4.1)

## Usuage

To run clustering followed by classification 

`python ConsumerClassifier/cluster_classify.py --dataset_path ConsumerClassifier/data/Technical\ test\ sample\ data.ods --output_dir model_store --ignore_features "consumer_id,gender,account_status,customer_age" `

To run model inferencing with new data

`python ConsumerClassifier/predict.py --output_dir output --model_path model_store/model.pkl --data_path ConsumerClassifier/data/test.csv `
