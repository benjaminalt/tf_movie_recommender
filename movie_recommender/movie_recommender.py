#!/bin/bash

import argparse

from database import tmdb_connector, imdb_connector
import os
import json

RESOURCES_DIR = os.path.join(os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir), "resources")
if not os.path.isdir(RESOURCES_DIR):
    os.makedirs(RESOURCES_DIR)


class MovieRecommender:
    """
    Will be used to store a data preprocessor and a TensorFlow model.
    Should be serializable for reuse.
    Should allow command-line interaction (possible queries: Random movie I could like, predicted top X, predict rating
        for given movie etc.)
    """
    def __init__(self):
        # Read TensorFlow model from file if provided
        raise NotImplementedError()

    def classify(self, title):
        """
        Rate a single IMDB movie (classify into one of ten categories).
        :return: The predicted rating (between 1 and 10 (inclusive))
        """
        raise NotImplementedError()


def train(update, database_backend="tmdb", labelled_movies_filepath=None, movie_info_filepath=None):
    # Get movie information (title, year, cast, director, ...)
    if movie_info_filepath:
        with open(movie_info_filepath) as movie_info_file:
            movie_info = json.load(movie_info_file)
    else:
        if database_backend == "imdb":
            db_connector = imdb_connector.IMDbConnector(labelled_movies_filepath)
        else:
            db_connector = tmdb_connector.TMDbConnector(os.path.join(RESOURCES_DIR, "credentials.json"))
        movie_info = db_connector.movie_info()
        with open(os.path.join(RESOURCES_DIR, "movie_info.json"), "w+") as dump_file:
            json.dump(movie_info, dump_file)

    # TODO: Train TensorFlow model
    # TODO: Start interactive MovieRecommender session


def classify():
    raise NotImplementedError()


def main(args):
    if args.command == "train":
        database = "imdb" if args.database == "imdb" else "tmdb"
        train(args.update, database, args.imdb_ratings, args.load_file)
    elif args.command == "classify":
        classify()
    else:
        raise Exception("Unknown command: {}".format(args.command))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="A movie recommendation system")
    parser.add_argument("command", type=str, help="train|classify")
    parser.add_argument("--update", action="store_true", help="Update the model with additional data.")
    parser.add_argument("--database", type=str, help="imdb|tmdb")
    parser.add_argument("--imdb_ratings", type=str, help="Path to CSV file containing movie titles and ratings (for IMDb "
                                                       "connector only).")
    parser.add_argument("--load_file", type=str, help="Path to JSON-formatted file containing movie information. If "
                                                      "this argument is provided, the movie information will not be "
                                                      "fetched from the database, but read from the given file.")
    parser.add_argument("--recent", action="store_true")
    main(parser.parse_args())