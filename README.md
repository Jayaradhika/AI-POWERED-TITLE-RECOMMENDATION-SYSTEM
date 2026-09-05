# AI-Powered Tile Recommendation System

Content-based recommendation engine that matches customer preferences
(room type, colour, style, finish, price range) to the best-fitting
tiles in a catalogue, using one-hot encoding + cosine similarity.

## Files
- `generate_dataset.py` — creates `data/tiles_dataset.csv` (300 synthetic tiles)
- `recommender.py` — `TileRecommender` class: cleaning, encoding, scaling, similarity, recommend()
- `main.py` — demo reproducing the slide's example (Living Room + White + Modern + Glossy + Medium)
- `evaluate.py` — Precision@K, Recall@K, F1-Score evaluation

## Run
```bash
pip install pandas numpy scikit-learn
python generate_dataset.py   # only needed once, or to regenerate data
python main.py               # get recommendations for a sample customer
python evaluate.py           # model evaluation metrics
```

## How it works
1. **Data Cleaning** — dedupe, strip whitespace, fill missing ratings
2. **Feature Selection** — Tile_Type, Room_Type, Colour, Size, Finish, Style, Price_Range, Rating, Usage
3. **Preprocessing** — `OneHotEncoder` for categorical columns, `MinMaxScaler` for Rating
4. **Similarity Calculation** — cosine similarity between the customer's preference vector and every tile's feature vector
5. **Recommendation** — top-N tiles ranked by Match_Score (%)

## Customize
Edit the `customer_preferences` dict in `main.py` to test different inputs.
Any key can be omitted — missing categorical fields are treated as "Unknown"
and missing Rating defaults to the dataset average.
