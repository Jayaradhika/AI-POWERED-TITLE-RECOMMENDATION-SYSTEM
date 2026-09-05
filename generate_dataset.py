"""
generate_dataset.py
--------------------
Creates a synthetic tile catalogue dataset with the exact feature schema
shown in the project slides (Tile Type, Room Type, Colour, Size, Finish,
Style, Price Range, Rating, Usage) and saves it to data/tiles_dataset.csv.
"""

import numpy as np
import pandas as pd

np.random.seed(42)

TILE_TYPES = ["Floor", "Wall", "Both"]
ROOM_TYPES = ["Living Room", "Bedroom", "Bathroom", "Kitchen", "Outdoor"]
COLOURS = ["White", "Beige", "Grey", "Black", "Brown", "Blue"]
SIZES = ["30x30 cm", "60x60 cm", "80x80 cm", "100x100 cm"]
FINISHES = ["Glossy", "Matte", "Satin", "Textured"]
STYLES = ["Modern", "Classic", "Rustic", "Minimalist"]
PRICE_RANGES = ["Low", "Medium", "High"]
USAGES = ["Indoor", "Outdoor", "Both"]

MATERIAL_BY_COLOUR = {
    "White": ["Ceramic", "Marble", "Porcelain"],
    "Beige": ["Sandstone", "Ceramic", "Travertine"],
    "Grey": ["Granite", "Concrete", "Porcelain"],
    "Black": ["Granite", "Slate", "Porcelain"],
    "Brown": ["Wood-look", "Terracotta", "Ceramic"],
    "Blue": ["Mosaic", "Ceramic", "Porcelain"],
}


def make_name(colour, style, material, finish):
    return f"{style} {colour} {material} ({finish})"


def generate(n_rows: int = 300) -> pd.DataFrame:
    rows = []
    for i in range(1, n_rows + 1):
        colour = np.random.choice(COLOURS)
        style = np.random.choice(STYLES)
        material = np.random.choice(MATERIAL_BY_COLOUR[colour])
        finish = np.random.choice(FINISHES)
        row = {
            "Tile_ID": f"T{i:04d}",
            "Tile_Name": make_name(colour, style, material, finish),
            "Tile_Type": np.random.choice(TILE_TYPES),
            "Room_Type": np.random.choice(ROOM_TYPES),
            "Colour": colour,
            "Size": np.random.choice(SIZES),
            "Finish": finish,
            "Style": style,
            "Price_Range": np.random.choice(PRICE_RANGES),
            "Rating": round(np.random.uniform(3.0, 5.0), 1),
            "Usage": np.random.choice(USAGES),
        }
        rows.append(row)
    return pd.DataFrame(rows)


if __name__ == "__main__":
    df = generate(300)
    df.to_csv("data/tiles_dataset.csv", index=False)
    print(f"Generated {len(df)} rows -> data/tiles_dataset.csv")
    print(df.head())
