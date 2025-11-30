import os
import glob
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

nltk.download('punkt')
nltk.download('stopwords')
nltk.download("punkt_tab")


stop_words = set(stopwords.words("english"))


def preprocess_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = text.lower()
    tokens = word_tokenize(text)
    tokens = [t for t in tokens if t.isalnum() and t not in stop_words]
    return " ".join(tokens)


def load_and_preprocess(data_folder="data") -> pd.DataFrame:
    """Loads ALL Datafiniti Amazon review CSV files, extracts needed columns."""

    csv_files = glob.glob(os.path.join(data_folder, "*.csv"))

    if not csv_files:
        raise FileNotFoundError("No .csv files found in /data folder!")

    print(f"Found {len(csv_files)} CSV files.")
    dfs = []

    for file in csv_files:
        try:
            df = pd.read_csv(file, low_memory=False)

            # Keep only columns we need
            col_map = {
                "name": "name",
                "reviews.text": "reviews.text",
                "reviews.rating": "reviews.rating",
            }

            missing = [c for c in col_map if c not in df.columns]
            if missing:
                print(f"WARNING: Missing columns {missing} in {file}. Skipping.")
                continue

            df = df[list(col_map.keys())]
            df.rename(columns=col_map, inplace=True)

            dfs.append(df)

        except Exception as e:
            print(f"Failed reading {file}: {e}")

    if not dfs:
        raise RuntimeError("No CSV files had the required review fields.")

    combined = pd.concat(dfs, ignore_index=True)
    print("Combined dataset size:", len(combined))

    # Clean text
    print("Cleaning text...")
    combined["clean_text"] = combined["reviews.text"].astype(str).apply(preprocess_text)

    return combined


if __name__ == "__main__":
    df = load_and_preprocess()
    print(df.head())
