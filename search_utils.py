import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Dict, Any, Tuple
import numpy as np


def create_full_text_for_startup(startup: Dict[str, Any]) -> str:
    """
    Creates a combined text from all searchable fields of a startup.
    """
    text_parts = []

    # Basic information
    text_parts.append(startup.get('name', ''))
    text_parts.append(startup.get('description', ''))

    # Canvas information (most important fields for search)
    canvas = startup.get('canvas', {})
    canvas_fields = [
        'Customer Segments', 'Problem', 'Unique Value Proposition',
        'Proposed Solution', 'Channels', 'Revenue Stream',
        'Unfair Advantage', 'Key Risks & Mitigation'
    ]

    for field in canvas_fields:
        value = canvas.get(field, '')
        if value and value != 'Not specified':
            text_parts.append(value)

    # Combine everything
    return ' '.join(filter(None, text_parts))


@st.cache_resource
def get_search_model(data: pd.DataFrame):
    """
    Creates and caches the TF-IDF model (Vectorizer and Matrix) for semantic search.
    Executed only once per session.
    """
    if 'full_text' not in data.columns:
        return None, None

    vectorizer = TfidfVectorizer(
        stop_words='english',
        ngram_range=(1, 2),
        min_df=2,
        max_features=5000
    )
    tfidf_matrix = vectorizer.fit_transform(data['full_text'])
    return vectorizer, tfidf_matrix


def semantic_search(query: str, startups: List[Dict[str, Any]], top_k: int = 10) -> List[Tuple[Dict[str, Any], float]]:
    """
    Performs semantic search across startup data.

    Args:
        query: User's search query
        startups: List of startup data
        top_k: Number of results to return

    Returns:
        List of (startup, score) tuples, sorted by relevance
    """
    if not query.strip() or not startups:
        return []

    try:
        # Create DataFrame with full_text column
        startup_data = []
        for startup in startups:
            startup_data.append({
                'id': startup.get('id'),
                'full_text': create_full_text_for_startup(startup),
                'startup_data': startup  # Original data for later
            })

        df = pd.DataFrame(startup_data)

        # Create TF-IDF model
        vectorizer, tfidf_matrix = get_search_model(df)

        if vectorizer is None or tfidf_matrix is None:
            return []

        # Transform query into TF-IDF space
        query_vector = vectorizer.transform([query])

        # Calculate Cosine Similarity
        similarity_scores = cosine_similarity(
            query_vector, tfidf_matrix).flatten()

        # Find Top-K results
        top_indices = np.argsort(similarity_scores)[::-1][:top_k]

        # Compile results (only those with Score > 0)
        results = []
        for idx in top_indices:
            score = similarity_scores[idx]
            if score > 0:  # Only relevant results
                startup = df.iloc[idx]['startup_data']
                results.append((startup, float(score)))

        return results

    except Exception as e:
        st.error(f"Error in semantic search: {e}")
        return []
