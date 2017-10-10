from __future__ import print_function
import tmdbsimple as tmdb
import json
import progressbar


class TMDbConnector(object):
    def __init__(self, credentials_filepath):
        with open(credentials_filepath) as credentials_file:
            self.credentials = json.load(credentials_file)
        print("Connecting to TMDb database...")
        tmdb.API_KEY = self.credentials["api_key"]
        auth = tmdb.Authentication()
        request_token = auth.token_new()["request_token"]
        res = auth.token_validate_with_login(request_token=request_token, username=self.credentials["username"], password=self.credentials["password"])
        if not res["success"]:
            raise RuntimeError("TMDb authentication failed")
        res = auth.session_new(request_token=request_token)
        if not res["success"]:
            raise RuntimeError("Could not create new TMDb session")
        self.account = tmdb.Account(session_id=res["session_id"])
        self.account.info()

        genres_list = tmdb.Genres().list()
        self.genres = {}
        for entry in genres_list["genres"]:
            self.genres[entry["id"]] = entry["name"]

        print("Done.")

    def movie_info(self):
        print("Fetching movie info...")
        res = []
        rated_movies = self.account.rated_movies()
        bar = progressbar.ProgressBar(max_value=rated_movies["total_results"])
        count = 0
        for i in range(rated_movies["total_pages"]):
            for movie in rated_movies["results"]:
                res.append(self.extract_movie_info(movie))
                bar.update(count)
                count += 1
            current_page = rated_movies["page"]
            if rated_movies["page"] != rated_movies["total_pages"]:
                rated_movies = self.account.rated_movies(page=current_page+1)
        return res

    def extract_movie_info(self, movie):
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
        credits = tmdb.Movies(movie["id"]).credits()
        cast = credits["cast"]
        if len(cast) > 10:
            cast = cast[:10]
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