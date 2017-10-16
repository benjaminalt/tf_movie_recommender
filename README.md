# tf_movie_recommender (alpha)

A machine learning-based movie recommendation system built with [TensorFlow](https://www.tensorflow.org/).
It uses a deep neural network to learn the user's movie preferences and predicts ratings for movies the user has not yet seen.

### Features
- Connect to a [TMDb](https://www.themoviedb.org/) account and use the list of rated movies as labelled input data
- Connect to [IMDb](http://www.imdb.com/) and use a CSV-formatted file containing labelled movies as input data
- Interactive command-line interface
- Predict movie ratings (between 0 and 9) for any movie that can be found on IMDb (or TMDb, depending on the backend)

### Dependencies
Due to some of the dependencies, this project requires Python 2.7. I installed the following into a dedicated virtualenv:
- [TensorFlow](https://www.tensorflow.org/) for Python 2.7
- [tmdbsimple](https://pypi.python.org/pypi/tmdbsimple)
- [imdbpy](http://imdbpy.sourceforge.net/)
- [pandas](http://pandas.pydata.org/)
- [progressbar2](https://pypi.python.org/pypi/progressbar2)
- [scikit-learn](http://scikit-learn.org/stable/)

### Usage
If applicable, activate the virtualenv, `cd` into the `movie_recommender` directory inside the repository and
launch `python movie_recommender.py`. The CLI will guide you through everything else.

### To Do
- Feature inference using an autoencoder. This can considerably reduce the dimensionality of the feature space and
improve prediction accuracy.
- Test and improve model serialization/deserialization.
- Functionality for updating the model incrementally: Do not retrain the classifier from scratch if a new movie was
added to the data set.
