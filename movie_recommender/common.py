from __future__ import print_function
import numpy as np


def make_feature_vector(movie_info):
    """
    Create a feature vector from detailed information about a movie.
    :param movie_info: Detailed information about a given movie
    :return: A feature vector
    """
    info = movie_info.data
    director = info["director"][0].data["name"]
    producer = info["producer"][0].data["name"]
    cast = [info["cast"][i].data["name"] for i in range(10)]
    genres = [info["genre"][i] for i in range(3)]
    rating = info["rating"]
    writer = info["writer"][0].data["name"]
    composer = info["original music"][0].data["name"]
    return np.array([director, producer, cast, genres, rating, writer, composer])