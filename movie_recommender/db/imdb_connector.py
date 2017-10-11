from __future__ import print_function
from imdb import IMDb
import progressbar
import pandas as pd


class IMDbConnector(object):
    """
    A db connector for obtaining movie information from the Internet Movie Database (https://www.imdb.com/).
    """
    def __init__(self, labelled_movies_filepath):
        """
        :param labelled_movies_filepath: A CSV-formatted file. Each row corresponds to a movie. Must have columns "id"
        and "rating", other columns are ignored.
        """
        self.labelled_movies = pd.read_csv(labelled_movies_filepath, sep=";", header=0)

    def movie_info(self):
        """
        Get relevant information about the set of rated movies.
        :return: A list of dicts, each of which contains information (such as genres, director, year...) about a movie
        """
        print("Getting movie info...")
        imdb_ids = self.labelled_movies["id"]
        ratings = self.labelled_movies["rating"]
        ia = IMDb()
        bar = progressbar.ProgressBar()
        res = []
        for index, movie_id in enumerate(bar(imdb_ids)):
            info = ia.get_movie(movie_id).data
            movie_dict = {
                "title": info["title"],
                "rating": ratings[index],
                "genres": info["genre"],
                "average_rating": info["rating"],
                "year": info["year"],
                "cast": [info["cast"][i].data["name"] for i in range(5 if len(info["cast"]) > 5 else len(info["cast"]))],
                "director": info["director"][0].data["name"],
                "producer": info["producer"][0].data["name"],
                "composer": info["original music"][0].data["name"],
                "writer": info["writer"][0].data["name"]
            }
            res.append(movie_dict)
        return pd.DataFrame(res)
