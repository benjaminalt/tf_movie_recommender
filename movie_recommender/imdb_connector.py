from __future__ import print_function
from imdb import IMDb
import progressbar

def movie_info(imdb_ids):
    """
    Get information about a set of movies.
    :param imdb_ids: A list of IMDb IDs of movies
    :return: A list containing detailed movie information about each of the given movies
    """
    print("Getting movie info...")
    ia = IMDb()
    bar = progressbar.ProgressBar()
    info = []
    for movie_id in bar(imdb_ids):
        info.append(ia.get_movie(movie_id))
    return info
