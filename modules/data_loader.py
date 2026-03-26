import pandas as pd
import numpy as np

def load_data():
    movies = pd.read_csv("ml-latest-small/movies.csv")
    ratings = pd.read_csv("ml-latest-small/ratings.csv")

    # Split genres into a list
    movies["genres"] = movies["genres"].apply(lambda x: x.split("|"))

    return movies, ratings

def get_genre_list(movies):
    all_genres = set()
    for genres in movies["genres"]:
        all_genres.update(genres)
    return sorted(list(all_genres))

def encode_movies(movies):
    """
    Convert each movie into a simple genre vector.
    This is our basic movie embedding for the prototype.
    """
    all_genres = get_genre_list(movies)

    def genre_vector(genres):
        return np.array([1.0 if g in genres else 0.0 for g in all_genres])

    movies["vector"] = movies["genres"].apply(genre_vector)
    return movies, all_genres

if __name__ == "__main__":
    movies, ratings = load_data()
    movies, genres = encode_movies(movies)

    print(f"Movies loaded: {len(movies)}")
    print(f"Ratings loaded: {len(ratings)}")
    print(f"Genres found: {genres}")
    print(f"\nSample movie vector ({movies.iloc[0]['title']}):")
    print(movies.iloc[0]["vector"])