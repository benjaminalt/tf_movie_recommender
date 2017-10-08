- **Training**
    - *Initial training*: Take list of movies with ratings between 1 and 10
        - Read in a CSV file with a list of movies and ratings
    - *Retraining*: When I watched a movie, want to update model with new rating 
      -> Incremental retraining
- **Predicting**
    - Once a week, give me a list of new movies which just came out which I could like
    - Classify the entire IMDB movie list and give me the 10s
   

Can use for training data only movies I have watched and labelled; use that information
to classify new movies into one of ten categories:

- **Input**:
    - Director
    - First 10 actors
    - Year
    - Scriptwriter
    - Score Composer
- **Output**:
    - Number between 1 and 10