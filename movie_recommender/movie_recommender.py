#!/bin/bash

import training
import imdb_connector
import common

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


def train(update, input_filepath):
    labelled_movies = pd.read_csv(input_filepath, sep=";", header=0)
    movie_info = imdb_connector.movie_info(labelled_movies["id"].tolist())
    feature_vectors = map(common.make_feature_vector, movie_info)
    model = training.train(feature_vectors, labelled_movies["rating"].tolist())


def main(args):
    if args.command == "train":
        if not args.input:
            raise ValueError("Argument '--input' required!")
        train(args.update, args.input)
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