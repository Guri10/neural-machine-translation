# scripts/preprocess_data.py
import os
import sys
# Add root directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
from sklearn.model_selection import train_test_split


INPUT_FILE = "data/raw/Sentence pairs in English-French - 2025-04-14.tsv"
OUTPUT_DIR = "data/processed"
MAX_SAMPLES = 200000

def load_and_clean(path, max_samples=None):
    df = pd.read_csv(path, sep='\t', header=None, names=['en_id', 'en', 'fr_id', 'fr'])

    # Basic text cleanup
    df['en'] = df['en'].astype(str).str.lower().str.strip()
    df['fr'] = df['fr'].astype(str).str.lower().str.strip()

    df = df[['en', 'fr']]  # Drop IDs
    df.dropna(inplace=True)
    df.drop_duplicates(inplace=True)

    if max_samples and len(df) > max_samples:
        df = df.sample(n=max_samples, random_state=42)

    return df

def split_and_save(df, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    train, test = train_test_split(df, test_size=0.1, random_state=42)
    train, val = train_test_split(train, test_size=0.1, random_state=42)

    train.to_csv(f"{output_dir}/train.csv", index=False)
    val.to_csv(f"{output_dir}/val.csv", index=False)
    test.to_csv(f"{output_dir}/test.csv", index=False)
    print("✅ Saved train/val/test to", output_dir)

if __name__ == "__main__":
    df = load_and_clean(INPUT_FILE, max_samples=MAX_SAMPLES)
    split_and_save(df, OUTPUT_DIR)
