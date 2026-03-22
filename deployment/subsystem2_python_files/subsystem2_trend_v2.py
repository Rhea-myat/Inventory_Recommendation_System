from pathlib import Path
import re
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

try:
    import kagglehub
except ImportError:
    kagglehub = None

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None

try:
    import hdbscan
except ImportError:
    hdbscan = None


BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "data" / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SOCIAL_TREND_OUTPUT = OUTPUT_DIR / "social_trend_signals_v2.csv"
CATEGORY_SOCIAL_OUTPUT = OUTPUT_DIR / "category_social_trends_v2.csv" 

ROOT_SOCIAL_TREND_OUTPUT = BASE_DIR / "social_trend_signals_v2.csv"
ROOT_CATEGORY_SOCIAL_OUTPUT = BASE_DIR / "category_social_trends_v2.csv"


def load_twitter_dataset() -> pd.DataFrame:
    if kagglehub is None:
        raise ImportError("kagglehub is not installed.")

    dataset_path = kagglehub.dataset_download(
        "mcantoni81/twitter-dataset-the-intel-raptor-release"
    )
    dataset_path = Path(dataset_path)

    csv_files = list(dataset_path.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError("No CSV file found in downloaded dataset.")

    tweet_path = csv_files[0]
    df = pd.read_csv(tweet_path)
    print("Loaded Twitter dataset:", tweet_path)
    print("Shape:", df.shape)
    return df


def clean_for_bert(text: str) -> str:
    if pd.isna(text):
        return ""
    text = str(text)
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    text = re.sub(r"@\w+", " ", text)
    text = re.sub(r"#", " ", text)
    text = re.sub(r"[^A-Za-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip().lower()
    return text


def cluster_topics(df: pd.DataFrame) -> pd.DataFrame:
    if SentenceTransformer is None or hdbscan is None:
        raise ImportError("sentence-transformers and hdbscan are required for topic clustering.")

    df = df.copy()
    df["clean_text"] = df["text"].apply(clean_for_bert)
    df = df.drop_duplicates(subset=["clean_text"]).reset_index(drop=True)

    texts = df["clean_text"].astype(str).tolist()

    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = embedding_model.encode(
        texts,
        batch_size=64,
        show_progress_bar=True,
        convert_to_numpy=True
    )

    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=15,
        min_samples=1,
        metric="euclidean"
    )

    labels = clusterer.fit_predict(embeddings)
    df["topic_id"] = labels
    return df


def top_words_per_topic(topic_docs: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:
    vectorizer = CountVectorizer(stop_words="english", min_df=1)
    X_counts = vectorizer.fit_transform(topic_docs["clean_text"])
    tfidf = TfidfTransformer()
    X_tfidf = tfidf.fit_transform(X_counts)
    terms = np.array(vectorizer.get_feature_names_out())

    rows = []
    for row_i, topic in enumerate(topic_docs["topic_id"].tolist()):
        row = X_tfidf[row_i].toarray().ravel()
        top_idx = row.argsort()[::-1][:top_n]
        keywords = terms[top_idx].tolist()

        rows.append({
            "topic_id": topic,
            "topic_keywords": keywords,
            "topic_keyword_text": " ".join(keywords)
        })

    return pd.DataFrame(rows)


def build_latest_topic_trends(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df_topics = df[df["topic_id"] != -1].copy()

    topic_docs = (
        df_topics.groupby("topic_id")["clean_text"]
        .apply(lambda s: " ".join(s.astype(str)))
        .reset_index()
    )

    topic_keywords_df = top_words_per_topic(topic_docs)

    df["created_at"] = pd.to_datetime(df["created_at"], errors="coerce")
    df = df.dropna(subset=["created_at"]).copy()
    df["period"] = df["created_at"].dt.date

    daily_mentions = (
        df[df["topic_id"] != -1]
        .groupby(["topic_id", "period"])
        .size()
        .reset_index(name="mentions_this_period")
        .sort_values(["topic_id", "period"])
    )

    daily_mentions["mentions_last_period"] = (
        daily_mentions.groupby("topic_id")["mentions_this_period"].shift(1)
    )

    m_last = daily_mentions["mentions_last_period"]
    daily_mentions["volume_growth"] = np.where(
        (m_last.isna()) | (m_last <= 0),
        0.0,
        (daily_mentions["mentions_this_period"] - m_last) / m_last
    )

    daily_mentions["novelty_score"] = daily_mentions["mentions_last_period"].isna().astype(int)

    if {"retweet_count", "like_count"}.issubset(df.columns):

        daily_engagement = (
            df[df["topic_id"] != -1]
            .groupby(["topic_id", "period"])[["retweet_count", "like_count"]]
            .mean()
            .reset_index()
        )

        # combine engagement
        daily_engagement["engagement_score"] = (
            daily_engagement["retweet_count"].fillna(0)
            + daily_engagement["like_count"].fillna(0)
        )

        daily_engagement = daily_engagement[
            ["topic_id", "period", "engagement_score"]
        ]

        daily_mentions = daily_mentions.merge(
            daily_engagement,
            on=["topic_id", "period"],
            how="left"
        )

    else:
        daily_mentions["engagement_score"] = daily_mentions["engagement_score"].fillna(0.0)

    daily_mentions["trend_score"] = (
        0.5 * daily_mentions["volume_growth"].clip(lower=0)
        + 0.3 * daily_mentions["engagement_score"]
        + 0.2 * daily_mentions["novelty_score"]
    )

    max_score = daily_mentions["trend_score"].max()
    if max_score > 0:
        daily_mentions["trend_score"] = daily_mentions["trend_score"] / max_score
        latest_topic_trends = (
        daily_mentions.sort_values(["topic_id", "period"])
        .groupby("topic_id", as_index=False)
        .tail(1)
        .merge(topic_keywords_df, on="topic_id", how="left")
    )

    latest_topic_trends["is_trending"] = (latest_topic_trends["trend_score"] > 0).astype(int)
    return latest_topic_trends


def load_subsystem1_output() -> pd.DataFrame:
    candidate_files = [
        BASE_DIR / "data" / "output" / "subsystem1_product_recommendations_v2.csv",
        BASE_DIR / "subsystem1_product_recommendations_v2.csv",
        BASE_DIR / "subsystem1_product_recommendations_v2.csv",
        Path("subsystem1_product_recommendations_v2.csv"),
        Path("alloc_out.csv"),
    ]
    for p in candidate_files:
        if p.exists():
            print("Loaded Subsystem 1 output from:", p)
            return pd.read_csv(p)

    raise FileNotFoundError(
        "Subsystem 1 output not found. Make sure subsystem1_product_recommendations.csv exists."
    )


STOP = {
    "intel", "amd", "asus", "asrock", "msi", "gigabyte", "supermicro",
    "corsair", "crucial", "oem", "tray", "gaming", "edition", "series",
    "gb", "tb", "ddr3", "ddr4", "ddr5", "x", "v", "ws", "pro", "plus"
}


def tokenize_name(text: str) -> set:
    text = str(text).lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    tokens = [t for t in text.split() if len(t) > 2 and t not in STOP and not t.isdigit()]
    return set(tokens)


def match_topics_to_categories(latest_topic_trends: pd.DataFrame, sub1_df: pd.DataFrame) -> pd.DataFrame:
    category_vocab = (
        sub1_df.groupby("CategoryName")["ProductName"]
        .apply(lambda s: set().union(*[tokenize_name(x) for x in s.astype(str)]))
        .to_dict()
    )

    for cat in sub1_df["CategoryName"].dropna().unique():
        category_vocab[cat] = category_vocab.get(cat, set()).union(tokenize_name(cat))

    topic_bridge = []

    for _, row in latest_topic_trends.iterrows():
        if isinstance(row.get("topic_keywords", []), list):
            topic_tokens = set(row["topic_keywords"])
        else:
            topic_tokens = set(str(row.get("topic_keyword_text", "")).split())

        if not topic_tokens:
            continue

        best_cat = None
        best_overlap = 0

        for cat, cat_tokens in category_vocab.items():
            overlap = len(topic_tokens.intersection(cat_tokens))
            if overlap > best_overlap:
                best_overlap = overlap
                best_cat = cat

        topic_bridge.append({
            "topic_id": row["topic_id"],
            "matched_category": best_cat,
            "keyword_overlap": best_overlap,
            "trend_score": row["trend_score"],
            "is_trending": row["is_trending"],
            "topic_keyword_text": row.get("topic_keyword_text", "")
        })

    topic_bridge_df = pd.DataFrame(topic_bridge)

    category_social = (
        topic_bridge_df.dropna(subset=["matched_category"])
        .sort_values(["matched_category", "trend_score"], ascending=[True, False])
        .drop_duplicates(subset=["matched_category"])
        .rename(columns={"matched_category": "CategoryName"})
        [["CategoryName", "trend_score", "is_trending", "topic_keyword_text", "keyword_overlap"]]
        .rename(columns={
            "trend_score": "social_trend_score",
            "is_trending": "social_is_trending"
        })
    )

    if category_social.empty:
        category_social = pd.DataFrame({
            "CategoryName": sorted(sub1_df["CategoryName"].dropna().unique()),
            "social_trend_score": 0.0,
            "social_is_trending": 0,
            "topic_keyword_text": "",
            "keyword_overlap": 0
        })

    return category_social


def load_existing_social_trends() -> pd.DataFrame:
    candidate_files = [
        SOCIAL_TREND_OUTPUT,
        ROOT_SOCIAL_TREND_OUTPUT,
        Path("social_trend_signals.csv"),
    ]
    for path in candidate_files:
        if path.exists():
            print("Loaded existing social trend signals from:", path)
            return pd.read_csv(path)

    raise FileNotFoundError(
        "social_trend_signals.csv not found, and fresh generation requires kagglehub, sentence-transformers, and hdbscan."
    )


if __name__ == "__main__":
    can_generate_social_trends = all(
        dependency is not None for dependency in (kagglehub, SentenceTransformer, hdbscan)
    )

    if can_generate_social_trends:
        twitter_df = load_twitter_dataset()
        clustered_df = cluster_topics(twitter_df)
        latest_topic_trends = build_latest_topic_trends(clustered_df)
    else:
        latest_topic_trends = load_existing_social_trends()

    latest_topic_trends.to_csv(SOCIAL_TREND_OUTPUT, index=False)
    print("Saved:", SOCIAL_TREND_OUTPUT)

    sub1_df = load_subsystem1_output()
    category_social = match_topics_to_categories(latest_topic_trends, sub1_df)
    category_social.to_csv(CATEGORY_SOCIAL_OUTPUT, index=False)
    category_social.to_csv(ROOT_CATEGORY_SOCIAL_OUTPUT, index=False)
    print("Saved:", CATEGORY_SOCIAL_OUTPUT)
