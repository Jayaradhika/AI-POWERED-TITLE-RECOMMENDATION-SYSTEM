"""
recommender.py
--------------
Content-Based Tile Recommendation engine.

Pipeline (matches the project's Machine Learning Methodology slide):
    Dataset -> Data Cleaning -> Feature Selection -> Data Preprocessing
    -> Model Training (fit encoders/scaler) -> Similarity Calculation
    -> Recommendation
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity

CATEGORICAL_FEATURES = [
    "Tile_Type", "Room_Type", "Colour", "Size",
    "Finish", "Style", "Price_Range", "Usage",
]
NUMERIC_FEATURES = ["Rating"]


class TileRecommender:
    def __init__(self):
        self.encoder = OneHotEncoder(handle_unknown="ignore")
        self.scaler = MinMaxScaler()
        self.df = None
        self.feature_matrix = None

    # ---------- Data Cleaning ----------
    @staticmethod
    def _clean(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df = df.drop_duplicates(subset="Tile_ID")
        for col in CATEGORICAL_FEATURES:
            df[col] = df[col].astype(str).str.strip()
        df["Rating"] = pd.to_numeric(df["Rating"], errors="coerce")
        df["Rating"] = df["Rating"].fillna(df["Rating"].mean())
        return df.reset_index(drop=True)

    # ---------- Model Training (fit transformers + build feature matrix) ----------
    def fit(self, df: pd.DataFrame):
        self.df = self._clean(df)

        cat_matrix = self.encoder.fit_transform(
            self.df[CATEGORICAL_FEATURES]
        ).toarray()
        num_matrix = self.scaler.fit_transform(self.df[NUMERIC_FEATURES])

        # categorical similarity matters more than the numeric rating,
        # so the rating column is down-weighted relative to the one-hot block
        self.feature_matrix = np.hstack([cat_matrix, num_matrix * 0.3])
        return self

    # ---------- Transform a single customer preference query ----------
    def _encode_query(self, preferences: dict) -> np.ndarray:
        query_row = {}
        for col in CATEGORICAL_FEATURES:
            query_row[col] = preferences.get(col, "Unknown")
        query_df = pd.DataFrame([query_row])
        cat_vec = self.encoder.transform(query_df[CATEGORICAL_FEATURES]).toarray()

        if "Rating" in preferences:
            rating_val = preferences["Rating"]
        else:
            rating_val = self.df["Rating"].mean()  # neutral assumption
        num_vec = self.scaler.transform(pd.DataFrame([[rating_val]], columns=NUMERIC_FEATURES))

        return np.hstack([cat_vec, num_vec * 0.3])

    # ---------- Recommendation ----------
    def recommend(self, preferences: dict, top_n: int = 5) -> pd.DataFrame:
        query_vec = self._encode_query(preferences)
        sims = cosine_similarity(query_vec, self.feature_matrix)[0]

        result = self.df.copy()
        result["Match_Score"] = np.round(sims * 100, 1)
        result = result.sort_values("Match_Score", ascending=False).head(top_n)
        cols = ["Tile_ID", "Tile_Name", "Room_Type", "Colour", "Style",
                "Finish", "Price_Range", "Rating", "Match_Score"]
        return result[cols].reset_index(drop=True)
