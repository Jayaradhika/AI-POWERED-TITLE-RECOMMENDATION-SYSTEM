"""
evaluate.py
-----------
Evaluates the recommender using a leave-one-out style test:
for each tile in the catalogue, its own attributes are used as a
"customer query", and we check whether tiles sharing its Room_Type +
Style (the "relevant" set) appear in the top-K recommendations.

Reports Precision@K, Recall@K, F1-Score, and an average Match_Score
as a proxy for user satisfaction.
"""

import numpy as np
import pandas as pd
from recommender import TileRecommender, CATEGORICAL_FEATURES


def evaluate(df: pd.DataFrame, k: int = 5, sample_size: int = 60):
    model = TileRecommender().fit(df)
    clean_df = model.df

    sample = clean_df.sample(n=min(sample_size, len(clean_df)), random_state=1)

    precisions, recalls, f1s, match_scores = [], [], [], []

    for _, tile in sample.iterrows():
        query = {col: tile[col] for col in CATEGORICAL_FEATURES}
        query["Rating"] = tile["Rating"]

        # "relevant" = same room type & style, excluding the query tile itself
        relevant_mask = (
            (clean_df["Room_Type"] == tile["Room_Type"]) &
            (clean_df["Style"] == tile["Style"]) &
            (clean_df["Tile_ID"] != tile["Tile_ID"])
        )
        relevant_ids = set(clean_df.loc[relevant_mask, "Tile_ID"])
        if not relevant_ids:
            continue

        recs = model.recommend(query, top_n=k + 1)
        recs = recs[recs["Tile_ID"] != tile["Tile_ID"]].head(k)
        recommended_ids = set(recs["Tile_ID"])

        hits = len(recommended_ids & relevant_ids)
        precision = hits / k
        recall = hits / min(len(relevant_ids), k)
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0

        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)
        match_scores.append(recs["Match_Score"].mean())

    return {
        "Precision@{}".format(k): round(np.mean(precisions), 3),
        "Recall@{}".format(k): round(np.mean(recalls), 3),
        "F1-Score": round(np.mean(f1s), 3),
        "Avg_Match_Score(%)": round(np.mean(match_scores), 1),
        "Samples_Evaluated": len(precisions),
    }


if __name__ == "__main__":
    df = pd.read_csv("data/tiles_dataset.csv")
    metrics = evaluate(df, k=5)
    print("Evaluation Metrics")
    print("-" * 30)
    for name, value in metrics.items():
        print(f"{name:22s}: {value}")
