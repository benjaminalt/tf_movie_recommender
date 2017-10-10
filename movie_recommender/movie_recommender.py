#!/bin/bash

import argparse

import pandas as pd

import common
import training
from database import tmdb_connector, imdb_connector
import os
import json

RESOURCES_DIR = os.path.join(os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir), "resources")
if not os.path.isdir(RESOURCES_DIR):
    os.makedirs(RESOURCES_DIR)


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


def train(update, database_backend="tmdb", input_filepath=None, movie_info_filepath=None):
    if movie_info_filepath:
        with open(movie_info_filepath) as movie_info_file:
            movie_info = json.load(movie_info_file)
    else:
        if database_backend == "imdb":
            labelled_movies = pd.read_csv(input_filepath, sep=";", header=0)
            db_connector = imdb_connector.IMDbConnector(labelled_movies)
        else:
            db_connector = tmdb_connector.TMDbConnector(os.path.join(RESOURCES_DIR, "credentials.json"))
        movie_info = db_connector.movie_info()
        with open(os.path.join(RESOURCES_DIR, "movie_info.json"), "w+") as dump_file:
            json.dump(movie_info, dump_file)
    feature_vectors = map(common.make_feature_vector, movie_info)
    model = training.train(feature_vectors, labelled_movies["rating"].tolist())


def main(args):
    if args.command == "train":
        if not args.input:
            raise ValueError("Argument '--input' required!")
        database = "imdb" if args.database == "imdb" else "tmdb"
        train(args.update, args.input. database)
    elif args.command == "classify":
        raise NotImplementedError()
    else:
        raise Exception("Unknown command: {}".format(args.command))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="A movie recommendation system")
    parser.add_argument("command", type=str, help="train|classify")
    parser.add_argument("--update", action="store_true")
    parser.add_argument("--database", type=str, help="imdb|tmdb")
    parser.add_argument("--imdb-input", type=str, help="Path to CSV file containing movie titles and ratings (for IMDb connector")
    parser.add_argument("--load-file", type=str, help="Path to json-formatted file containing movie information")
    parser.add_argument("--recent", action="store_true")
    main(parser.parse_args())