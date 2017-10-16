#!/bin/bash
from __future__ import print_function

from db import tmdb_connector, imdb_connector
from ml.data_preprocessor import DataPreprocessor
from ml.dnn import DNN, TMP_MODEL_DIR

import tempfile, shutil
import cmd
import os
import json
import pickle, tarfile
import pandas as pd
from sklearn.model_selection import train_test_split

RESOURCES_DIR = os.path.join(os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir), "resources")
if not os.path.isdir(RESOURCES_DIR):
    os.makedirs(RESOURCES_DIR)


class MovieRecommender(cmd.Cmd):
    """
    An interactive movie recommender system which learns the user's preferences.
    """
    def __init__(self):
        cmd.Cmd.__init__(self)
        self.movie_info = None
        self.db_connector = None
        self.data_preprocessor = None
        self.classifier = None

    def do_train(self, line):
        """Train the model."""
        use_input_file = raw_input("Would you like me to load the movie metadata from a file? [y|N] ")
        if use_input_file.lower() in ["y", "yes"]:
            self.movie_info = self.__load_movie_info_from_file()
        else:
            self.db_connector = self.__get_db_connector()
            if self.db_connector is None:
                return
            self.movie_info = self.db_connector.movie_info()
        if self.movie_info is None:
            print("Loading movie metadata failed.")
            return
        print("Movie info loaded.")
        dump_movie_info = raw_input("Would you like me to save the gathered movie information to a file for future use? [Y|n] ")
        if dump_movie_info.lower() in ["", "y", "yes"]:
           self.__dump_movie_info(self.movie_info)
        self.data_preprocessor = DataPreprocessor(self.movie_info)
        encoded = pd.DataFrame(self.movie_info.apply(self.data_preprocessor.encode, axis=1).tolist())
        df_train, df_test = train_test_split(encoded, test_size=0.1)
        self.classifier = DNN()
        self.classifier.train(df_train, df_test)

    def do_load(self, line):
        filename = raw_input("Path to archive file: ")
        extract_dir = raw_input("Extract archive to: ")
        if os.path.exists(filename) and os.path.isfile(filename) and filename.endswith(".tar.gz")\
                and os.path.exists(extract_dir) and os.path.isdir(extract_dir):
            tar = tarfile.open(filename)
            tar.extractall(extract_dir)
            archive_dir = os.path.join(extract_dir, "movie_recommender_archive")
            model_dir = os.path.join(archive_dir, "model_archive")
            with open(os.path.join(archive_dir, "feature_columns.pickle"), "r") as feature_columns_file:
                feature_columns = pickle.load(feature_columns_file)
            self.classifier = DNN(feature_columns=feature_columns, model_dir=model_dir)
            with open(os.path.join(archive_dir, "data_preprocessor.pickle"), "r") as preprocessor_file:
                self.data_preprocessor = pickle.load(preprocessor_file)
            print("Model loaded successfully.")
        else:
            print("Could not load model: Archive does not exist or is invalid")

    def do_save(self, line):
        if self.data_preprocessor is None or self.classifier is None:
            print("Cannot save: No data preprocessor and classifier exist")
            return
        filename = raw_input("Path to archive file (.tar.gz): ")
        if not filename.endswith(".tar.gz"):
            print("Could not save model to file: Invalid extension")
            return
        temp_dir = tempfile.mkdtemp()
        shutil.copytree(TMP_MODEL_DIR, os.path.join(temp_dir, "model_archive"))
        with open(os.path.join(temp_dir, "data_preprocessor.pickle"), "w+") as data_preprocessor_file:
            pickle.dump(self.data_preprocessor, data_preprocessor_file)
        with open(os.path.join(temp_dir, "feature_columns.pickle"), "w+") as feature_columns_file:
            pickle.dump(self.classifier.feature_columns, feature_columns_file)
        tar = tarfile.open(filename, "w:gz")
        tar.add(temp_dir, arcname="movie_recommender_archive")
        tar.close()
        shutil.rmtree(temp_dir)
        print("Model saved.")

    def do_update(self, line):
        """Update the model."""
        pass

    def do_predict(self, line):
        """Predict a rating for a movie."""
        if self.classifier is None or self.data_preprocessor is None:
            print("No classifier loaded. Train or load a classifier before predicting.")
            return
        if self.db_connector is None:
            self.db_connector = self.__get_db_connector()
        if self.db_connector is None:
            print("Setting database connector failed. Aborting prediction.")
            return
        movie_info = self.__get_movie()
        encoded = self.data_preprocessor.encode(movie_info)
        encoded_df = pd.DataFrame([encoded])
        predicted_rating = self.classifier.predict(encoded_df)
        print("Predicted rating: {}".format(predicted_rating))

    def do_exit(self, line):
        return True

    @staticmethod
    def __get_db_connector():
        """
        Prompt the user for his/her preferred database backend.
        :return: The database connector on success, None on failure
        """
        inp = raw_input("Which database backend would you like to use? Available: tmdb | imdb [tmdb] ")
        if inp.lower() in ["", "tmdb"]:
            filepath = raw_input("Path to JSON-formatted credentials file with entries 'api_key', 'username', 'password': ")
            if not (os.path.exists(filepath) and os.path.isfile(filepath)):
                print("File {} does not exist".format(filepath))
                return None
            return tmdb_connector.TMDbConnector(filepath)
        elif inp.lower() == "imdb":
            filepath = raw_input("Path to CSV-formatted input file with IMDb IDs and movie ratings: ")
            if not (os.path.exists(filepath) and os.path.isfile(filepath)):
                print("File {} does not exist".format(filepath))
                return None
            return imdb_connector.IMDbConnector(filepath)
        else:
            print("Invalid input: {}".format(inp))
            return None

    @staticmethod
    def __load_movie_info_from_file():
        filename = raw_input("Path to JSON-formatted input file: ")
        if not(os.path.exists(filename) and os.path.isfile(filename)):
            print("File {} does not exist".format(filename))
            return None
        try:
            with open(filename) as movie_info_file:
                return pd.DataFrame(json.load(movie_info_file))
        except Exception as e:
            print("Could not read input file")
            return None

    @staticmethod
    def __dump_movie_info(movie_info):
        filename = os.path.join(RESOURCES_DIR, "movie_info.json")
        res = raw_input("Shall I save the movie information to {}? [Y|n] ".format(filename))
        if res.lower() in ["n", "no"]:
            filename = raw_input("Alternate file path: ")
        with open(filename, "w+") as dump_file:
            movie_info.to_json(dump_file)
        print("Movie information saved.")

    def __get_movie(self):
        movie = None
        while movie is None:
            movie_title = raw_input("Movie title: ")
            movie_list = self.db_connector.movie_list(movie_title)
            if len(movie_list) == 0:
                try_again = raw_input("Could not find a movie with title '{}'. Try again? [Y|n] ".format(movie_title))
                if try_again.lower() in ["", "yes", "y"]:
                    continue
                break
            print("Found movies:")
            print("{:>4} {:>30} {:>20}".format("#", "Title", "Release Date"))
            print("--------------------------------------------------------")
            for index, found_movie in enumerate(movie_list):
                print("{:>3} {:>20} {:>10}".format(index, found_movie["title"], found_movie["release_date"]))
            index = raw_input("Use movie (index, q to search again): [0]")
            if not index.isdigit() or int(index) > len(movie_list) or int(index) < 0:
                try_again = raw_input("Aborted. Try again? [Y|n] ")
                if try_again.lower() in ["", "yes", "y"]:
                    continue
                break
            movie = self.db_connector.movie_info_from_id(movie_list[int(index)]["id"])
        return movie


if __name__ == "__main__":
    MovieRecommender().cmdloop(intro="""A machine learning-powered movie recommendation system.
    Available commands: train | predict | save | load | exit""")