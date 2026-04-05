import urllib.request
import urllib.parse
import json
import os
import time
from dotenv import load_dotenv

load_dotenv()

# Try Streamlit secrets first, then .env
API_KEY = None
try:
    import streamlit as st
    API_KEY = st.secrets.get("OMDB_API_KEY")
except Exception:
    pass

if not API_KEY:
    API_KEY = os.getenv("OMDB_API_KEY")

BASE_URL = "http://www.omdbapi.com/"

# Simple in-memory cache to avoid hitting API limits
_cache = {}

def get_movie_data(title, year=None):
    """
    Fetch movie metadata from OMDB by title.
    Returns dict with plot, poster, rating, cast etc.
    """
    cache_key = f"{title}_{year}"
    if cache_key in _cache:
        return _cache[cache_key]

    params = {"t": title, "apikey": API_KEY, "plot": "short"}
    if year:
        params["y"] = year

    url = BASE_URL + "?" + urllib.parse.urlencode(params)

    try:
        response = urllib.request.urlopen(url, timeout=5)
        data = json.loads(response.read())

        if data.get("Response") == "True":
            result = {
                "title":    data.get("Title", title),
                "year":     data.get("Year", ""),
                "plot":     data.get("Plot", "No description available."),
                "poster":   data.get("Poster", "N/A"),
                "rating":   data.get("imdbRating", "N/A"),
                "director": data.get("Director", "N/A"),
                "cast":     data.get("Actors", "N/A"),
                "runtime":  data.get("Runtime", "N/A"),
                "language": data.get("Language", "N/A"),
            }
        else:
            result = _empty_result(title)

    except Exception:
        result = _empty_result(title)

    _cache[cache_key] = result
    return result


def _empty_result(title):
    return {
        "title":    title,
        "year":     "",
        "plot":     "No description available.",
        "poster":   "N/A",
        "rating":   "N/A",
        "director": "N/A",
        "cast":     "N/A",
        "runtime":  "N/A",
        "language": "N/A",
    }


def clean_title(raw_title):
    """
    Convert MovieLens title format to something OMDB can find.
    Examples:
      'Dark Knight, The (2008)' → ('The Dark Knight', '2008')
      'Toy Story (1995)'        → ('Toy Story', '1995')
      'Matrix, The'             → ('The Matrix', None)
    """
    title = raw_title.strip()
    year  = None

    # Extract year
    if "(" in title and ")" in title:
        try:
            potential_year = title[title.rfind("(")+1:title.rfind(")")]
            if potential_year.isdigit() and len(potential_year) == 4:
                year  = potential_year
                title = title[:title.rfind("(")].strip()
        except Exception:
            pass

    # Fix "Title, The" → "The Title"
    for article in [", The", ", A", ", An"]:
        if title.endswith(article):
            title = article.strip(", ") + " " + title[:-len(article)]
            break

    return title.strip(), year


def enrich_recommendations(recommendations, delay=0.1):
    """
    Take a list of recommendation dicts and add OMDB metadata to each.
    Uses a small delay to avoid hitting rate limits.
    """
    enriched = []
    for rec in recommendations:
        title, year = clean_title(rec["title"])
        omdb = get_movie_data(title, year)

        # If no poster found, try without year
        if omdb["poster"] == "N/A" and year:
            omdb = get_movie_data(title, None)

        enriched.append({**rec, **omdb})
        time.sleep(delay)

    return enriched


if __name__ == "__main__":
    test_movies = [
        {"title": "Inception (2010)",        "score": 0.95},
        {"title": "Dark Knight, The (2008)", "score": 0.91},
        {"title": "Toy Story (1995)",        "score": 0.88},
        {"title": "Matrix, The (1999)",      "score": 0.85},
        {"title": "Schindler's List (1993)", "score": 0.82},
    ]

    results = enrich_recommendations(test_movies)
    for r in results:
        print(f"\n{r['title']} ({r['year']})")
        print(f"  Rating:   {r['rating']}")
        print(f"  Director: {r['director']}")
        print(f"  Cast:     {r['cast']}")
        print(f"  Plot:     {r['plot'][:80]}...")
        print(f"  Poster:   {'✅' if r['poster'] != 'N/A' else '❌'}")