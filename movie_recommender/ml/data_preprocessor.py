from __future__ import print_function, division
import utils
import pandas as pd

class DataPreprocessor(object):
    def __init__(self, movie_info):
        self.known_values = {
            "actor": self.distinct_from_list_valued_column("cast", movie_info),
            "genre": self.distinct_from_list_valued_column("genres", movie_info),
            "director": list(set(movie_info["director"].tolist())),
            "composer": list(set(movie_info["composer"].tolist())),
            "producer": list(set(movie_info["producer"].tolist())),
            "writer": list(set(movie_info["writer"].tolist()))
        }
        self.extrema = { "min": movie_info.min(numeric_only=True), "max": movie_info.max(numeric_only=True) }

    @staticmethod
    def distinct_from_list_valued_column(column_name, df):
        res = set()
        for idx, row in df.iterrows():
            for item in row[column_name]:
                res.add(item)
        return res

    @staticmethod
    def normalize_categorical(string):
        return string if (utils.is_ascii(string) and len(string) > 0) else "Unknown"

    @staticmethod
    def normalize_continuous(val, min, max):
        return (val - min) / (max - min)

    def encode(self, movie):
        """
        Combination of one-hot encoding for categorical variables and scaled between 0 and 1 for continuous variables
        :param movie:
        :return:
        """
        encoded = {
            "year": self.normalize_continuous(movie["year"], self.extrema["min"].get("year"), self.extrema["max"].get("year")),
            "average_rating": self.normalize_continuous(movie["average_rating"], self.extrema["min"].get("average_rating"), self.extrema["max"].get("average_rating")),
        }
        if "rating" in movie.keys(): # Not necessary for "predict" input data (which does not have a rating yet)
            encoded["rating"] = int(movie["rating"]) - 1  # Do not normalize the rating, as it serves as a class label and must be integer-valued (for 10 classes, between 0 and 9)
        for actor in self.known_values["actor"]:
            normalized = self.normalize_categorical(actor)
            encoded["actor_{}".format(normalized)] = 1 if actor in movie["cast"] else 0
        for genre in self.known_values["genre"]:
            normalized = self.normalize_categorical(genre)
            encoded["genre_{}".format(normalized)] = 1 if genre in movie["genres"] else 0
        for category in ["director", "writer", "producer", "composer"]:
            for item in self.known_values[category]:
                normalized = self.normalize_categorical(item)
                encoded["{}_{}".format(category, normalized)] = 1 if item in movie[category] else 0
        return encoded

    def make_feature_vector(self, movie):
        pass

    def make_one_hot_encoding(self, movie_info):
        encoded = {
            "year": movie_info["year"].tolist(),
            "average_rating": movie_info["average_rating"].tolist()
        }
        row_count = movie_info.shape[0]
        for numeric_column in ["year", "average_rating"]:
            encoded[numeric_column] = movie_info[numeric_column].tolist()
        for index, movie in movie_info.iterrows():
            for list_like_category in ["genres", "cast"]:
                for item in movie[list_like_category]:
                    item = "Unknown" if (item == "" or not utils.is_ascii(item)) else item
                    one_hot_colname = "{}_{}".format(list_like_category, item)
                    if one_hot_colname not in encoded.keys():
                        encoded[one_hot_colname] = [0] * row_count
                    encoded[one_hot_colname][int(index)] = 1
            for single_category in ["producer", "writer", "composer"]:
                item = "Unknown" if (movie[single_category] == "" or not utils.is_ascii(movie[single_category])) else movie[single_category]
                one_hot_colname = "{}_{}".format(single_category, item)
                if one_hot_colname not in encoded.keys():
                    encoded[one_hot_colname] = [0] * row_count
                encoded[one_hot_colname][int(index)] = 1
        return pd.DataFrame(encoded)