"""
main.py
-------
Demo entry point: loads the dataset, trains the content-based
recommender, and reproduces the "Customer Input -> AI Recommendations"
example from the project slides (Living Room + White + Modern + Glossy
+ Medium Budget).
"""

import pandas as pd
from recommender import TileRecommender


def main():
    df = pd.read_csv("data/tiles_dataset.csv")
    model = TileRecommender().fit(df)

    customer_preferences = {
        "Room_Type": "Living Room",
        "Colour": "White",
        "Style": "Modern",
        "Finish": "Glossy",
        "Price_Range": "Medium",
    }

    print("CUSTOMER INPUT")
    print(" + ".join(customer_preferences.values()))
    print()

    recommendations = model.recommend(customer_preferences, top_n=5)
    print("AI RECOMMENDATIONS")
    print(recommendations.to_string(index=False))


if __name__ == "__main__":
    main()
