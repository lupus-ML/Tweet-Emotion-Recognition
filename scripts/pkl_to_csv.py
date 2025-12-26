from pathlib import Path
import json
import pandas as pd


def main():
    """
    Convert merged_training.pkl (pandas DataFrame) into a clean CSV for notebooks/ML,
    and save a label mapping JSON for reproducibility.
    """

    # Defining project root and file paths
    project_root = Path(__file__).resolve().parents[1]
    raw_path = project_root / "data" / "raw" / "merged_training.pkl"
    out_csv_path = project_root / "data" / "processed" / "emotion_dataset.csv"
    out_map_path = project_root / "data" / "processed" / "label_mapping.json"

    # Error handling for missing input
    if not raw_path.exists():
        raise FileNotFoundError(f"Input file not found: {raw_path}")

    # Loading DataFrame
    df = pd.read_pickle(raw_path)

    # Validating DataFrame
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Pickle did not contain a pandas DataFrame.")
    if df.empty:
        raise ValueError("The DataFrame is empty.")

    # Printing basic info 
    print("Project root:", project_root)
    print("Raw path:", raw_path)
    print("Output CSV path:", out_csv_path)
    print("Shape:", df.shape)
    print("Columns:", df.columns.tolist())
    print(df.head(5))

    # Detecting text + label columns 
    possible_text_cols = ["text", "sentence", "content"]
    possible_emotion_cols = ["emotion", "emotions", "label", "labels"]

    text_col = next((c for c in possible_text_cols if c in df.columns), None)
    emotion_col = next((c for c in possible_emotion_cols if c in df.columns), None)

    if text_col is None:
        raise KeyError(f"Could not find a text column. Looked for: {possible_text_cols}")
    if emotion_col is None:
        raise KeyError(f"Could not find an emotion/label column. Looked for: {possible_emotion_cols}")

    # Keep only needed columns and standardize names
    df = df[[text_col, emotion_col]].copy()
    df.rename(columns={text_col: "text", emotion_col: "emotion"}, inplace=True)

    # Cleaning values
    df["text"] = df["text"].astype(str).str.strip()
    df["emotion"] = df["emotion"].astype(str).str.strip().str.lower()

    # Removing empty texts
    before = len(df)
    df = df[df["text"].str.len() > 0].copy()
    print(f"Removed {before - len(df)} empty text rows.")

    # Create numeric labels for training (stable mapping)
    emotions_sorted = sorted(df["emotion"].unique())
    emotion_to_id = {emo: i for i, emo in enumerate(emotions_sorted)}
    df["label"] = df["emotion"].map(emotion_to_id).astype(int)

    # Printing distribution + mapping (important for debugging & reporting)
    print("\nEmotion distribution:")
    print(df["emotion"].value_counts())

    print("\nEmotion -> label mapping:")
    print(emotion_to_id)

    # Ensuring output folder exists
    out_csv_path.parent.mkdir(parents=True, exist_ok=True)

    # Save 
    df.to_csv(out_csv_path, index=False, encoding="utf-8")
    with open(out_map_path, "w", encoding="utf-8") as f:
        json.dump(emotion_to_id, f, indent=2)

    print(f"\nSaved CSV to: {out_csv_path}")
    print(f"Saved label mapping to: {out_map_path}")


if __name__ == "__main__":
    main()