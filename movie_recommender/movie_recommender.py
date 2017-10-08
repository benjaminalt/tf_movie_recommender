#!/bin/bash

import training
import argparse
import pandas as pd

class MovieRecommender:
    def __init__(self):
        # Read TensorFlow model from file
        raise NotImplementedError()

    def classify(self, title):
        """
        Classify a single IMDB movie into one of ten categories.
        :return:
        """
        raise NotImplementedError()

def main(args):
    if args.command == "train":
        if not args.input:
            raise ValueError("Argument '--input' required!")
        if args.update:
            raise NotImplementedError()
        else:
            labelled_inputs = pd.read_csv(args.input)
            model = training.train(labelled_inputs)
    elif args.command == "classify":
        raise NotImplementedError()
    else:
        raise Exception("Unknown command: {}".format(args.command))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="A movie recommendation system")
    parser.add_argument("command", type=str, help="train|classify")
    parser.add_argument("--update", action="store_true")
    parser.add_argument("--input", type=str, help="Path to CSV file containing movie titles and ratings")
    parser.add_argument("--recent", action="store_true")
    main(parser.parse_args())