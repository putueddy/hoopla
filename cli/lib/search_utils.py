import json
from pathlib import Path

DEFAULT_SEARCH_LIMIT = 5
BM25_K1 = 1.5
BM25_B = 0.75
DEFAULT_CHUNK_SIZE = 200

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "movies.json"
STOPWORDS_PATH = PROJECT_ROOT / "data" / "stopwords.txt"
CACHE_DIR = PROJECT_ROOT / "cache"


def load_movies() -> list[dict]:
    """Load movie data from data/movies.json"""
    try:
        with open(DATA_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
            return data.get('movies', [])
    except FileNotFoundError:
        print(f"movies.json not found at: {DATA_PATH}")
    except json.JSONDecodeError as e:
        print(f"Failed to parse movies.json: {e}")
    except Exception as e:
        print(f"Failed to load movies: {e}")
    return []

def load_stop_words() -> list[str]:
    """Load stopword list from data/stopwords.txt"""
    try:
        with open(STOPWORDS_PATH, "r", encoding="utf-8") as f:
            return f.read().splitlines()
    except FileNotFoundError:
        print(f"stopwords.txt not found at: {STOPWORDS_PATH}")
    except Exception as e:
        print(f"Failed to load stopwords: {e}")
    return []