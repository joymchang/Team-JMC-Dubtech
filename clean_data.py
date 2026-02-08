import pandas as pd


INPUT_PATH = "data/Access_to_Tech_Dataset.csv"
OUTPUT_PATH = "data/Access_to_Tech_Dataset_cleaned.csv"


def _normalize_domain_category(series: pd.Series) -> pd.Series:
    normalized = series.astype(str).str.strip()

    #Consolidate Domains
    normalized = normalized.replace(
        {
            "TechnologyScienceResearch": "Technology Science and Research",
            "Ecommerce": "E-commerce",
        }
    )

    # Streaming to Tech
    normalized = normalized.replace(
        {
            "Streaming Platforms": "Technology Science and Research",
            "Streaming": "Technology Science and Research",
        }
    )

    return normalized


def _strip_string_columns(df: pd.DataFrame) -> pd.DataFrame:
    for col in df.columns:
        if pd.api.types.is_string_dtype(df[col]):
            df[col] = df[col].str.strip()
    return df


def main() -> None:
    df = pd.read_csv(INPUT_PATH)

    df = _strip_string_columns(df)

    if "domain_category" in df.columns:
        df["domain_category"] = _normalize_domain_category(df["domain_category"])


    df = df.drop_duplicates()


    dedupe_cols = [
        col
        for col in [
            "web_URL",
            "violation_name",
            "violation_description",
            "violation_category",
            "violation_impact",
        ]
        if col in df.columns
    ]
    if dedupe_cols:
        df = df.drop_duplicates(subset=dedupe_cols)

    df.to_csv(OUTPUT_PATH, index=False)
    print(f"Saved cleaned data to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
