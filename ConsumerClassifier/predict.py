import argparse
from consumer_classifier import classifier

if __name__ =='__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model_path', type=str)
    parser.add_argument('-d', '--data_path', type=str)
    parser.add_argument('-o', '--output_dir', type=str)
    args = parser.parse_args()

    classifier = classifier(args.model_path, args.data_path, args.output_dir)
    classifier.predict()
