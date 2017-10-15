from __future__ import print_function
import tmdbsimple as tmdb
import json
import progressbar
import pandas as pd


class TMDbConnector(object):
    """
    A db connector for obtaining movie information from The Movie Database (https://www.themoviedb.org/).
    """
    def __init__(self, credentials_filepath):
        """
        :param credentials_filepath: Path to JSON file containing entries for keys "api_key", "username" and "password",
        which are required for connecting to TMDb and obtaining a list of rated movies from a TMDb account.
        """
        with open(credentials_filepath) as credentials_file:
            self.credentials = json.load(credentials_file)
        self.authenticate()
        genres_list = tmdb.Genres().list()
        self.genres = {}
        for entry in genres_list["genres"]:
            self.genres[entry["id"]] = entry["name"]

        print("Done.")

    def authenticate(self):
        print("Connecting to TMDb db...")
        tmdb.API_KEY = self.credentials["api_key"]
        auth = tmdb.Authentication()
        request_token = auth.token_new()["request_token"]
        res = auth.token_validate_with_login(request_token=request_token, username=self.credentials["username"],
                                             password=self.credentials["password"])
        if not res["success"]:
            raise RuntimeError("TMDb authentication failed")
        res = auth.session_new(request_token=request_token)
        if not res["success"]:
            raise RuntimeError("Could not create new TMDb session")
        self.account = tmdb.Account(session_id=res["session_id"])
        self.account.info()

    def movie_info(self):
        """
        Get relevant information about the set of rated movies. Fetches the set of rated movies from the TMDb account.
        :return: A DataFrame containing information (such as genres, director, year...) about the movie.
        """
        print("Fetching movie info...")
        res = []
        rated_movies = self.account.rated_movies()
        bar = progressbar.ProgressBar(max_value=rated_movies["total_results"])
        count = 0
        for i in range(rated_movies["total_pages"]):
            for movie in rated_movies["results"]:
                res.append(self.__extract_movie_info(movie))
                bar.update(count)
                count += 1
            current_page = rated_movies["page"]
            if rated_movies["page"] != rated_movies["total_pages"]:
                rated_movies = self.account.rated_movies(page=current_page+1)
        return pd.DataFrame(res)

    def movie_info_from_id(self, movie_id):
        """
        Get relevant information about a given movie. (Returns a row of the dataframe returned by the other movie_info
        function, except for without the rating column (as the movie has not necessarily been rated yet)
        :param movie_id: An integer TMDB movie ID.
        :return:
        """
        movie = tmdb.Movies(movie_id)
        response = movie.info()
        movie_dict = {
            "title": movie.title,
            "genres": [entry["name"] for entry in movie.genres],
            "average_rating": movie.vote_average,
            "year": int(movie.release_date[:4]),
            "cast": [],
            "director": "",
            "producer": "",
            "composer": "",
            "writer": ""
        }
        return self.__add_cast_and_crew(movie_id, movie_dict)

    @staticmethod
    def __extract_movie_info(movie):
        """
        Fill a dictionary object with information about a movie.
        :param movie:
        :return:
        """
        movie_dict = {
            "title": movie["title"],
            "rating": movie["rating"],
            "genres": [self.genres[k] for k in movie["genre_ids"]],
            "average_rating": movie["vote_average"],
            "year": int(movie["release_date"][:4]),
            "cast": [],
            "director": "",
            "producer": "",
            "composer": "",
            "writer": ""
        }
        return TMDbConnector.__add_cast_and_crew(movie["id"], movie_dict)

    @staticmethod
    def __add_cast_and_crew(movie_id, movie_dict):
        credits = tmdb.Movies(movie_id).credits()
        cast = credits["cast"]
        if len(cast) > 5:
            cast = cast[:5]
        for actor in cast:
            movie_dict["cast"].append(actor["name"])
        crew = credits["crew"]
        for crew_member in crew:
            job = crew_member["job"]
            if job == "Director" and movie_dict["director"] == "":
                movie_dict["director"] = crew_member["name"]
            elif job == "Producer" and movie_dict["producer"] == "":
                movie_dict["producer"] = crew_member["name"]
            elif job == "Original Music Composer" and movie_dict["composer"] == "":
                movie_dict["composer"] = crew_member["name"]
            elif job in ["Writer", "Novel"] and movie_dict["writer"] == "":
                movie_dict["writer"] = crew_member["name"]
        return movie_dict

    @staticmethod
    def movie_list(movie_title):
        search = tmdb.Search()
        response = search.movie(query=movie_title)
        return search.results