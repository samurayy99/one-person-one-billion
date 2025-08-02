import streamlit as st
from typing import Dict, Any, List
# Stelle sicher, dass diese Dateien existieren und die Funktionen korrekt sind
from views.dossier_view import dossier_view
from data_utils import load_startup_data

# Load external CSS file
with open('styles.css') as f:
    st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)


def display_startup_grid(startups: List[Dict[str, Any]]):
    """
    Renders a clean grid with native Streamlit components.
    All startups are displayed with search and filter functionality.
    """
    st.title("One-Person, One-Billion Dollar Company")
    st.header("A Strategic Report on the Future of Solo Entrepreneurship, Modular AI, and Department-Free Organizations")
    st.markdown("**Engin Caglar**")

    # Search and Filter Section - Compact Version
    with st.container():
        # Create two columns for search
        search_col1, search_col2 = st.columns(2)

        with search_col1:
            # Sector/Industry Filter
            all_industries = list(
                set([startup.get('industry_focus', 'Other') for startup in startups]))
            all_industries.sort()
            selected_industry = st.selectbox(
                "🔍 Filter by Industry/Sector:",
                ["All Industries"] + all_industries,
                key="industry_filter"
            )

        with search_col2:
            # Keyword Search
            search_query = st.text_input(
                "🔎 Search by Keywords:",
                placeholder="Enter keywords like 'AI', 'healthcare', 'fintech'...",
                key="keyword_search"
            )

        # Apply filters
        filtered_startups = startups.copy()

        # Filter by industry
        if selected_industry != "All Industries":
            filtered_startups = [s for s in filtered_startups if s.get(
                'industry_focus', 'Other') == selected_industry]

        # Filter by keyword search
        if search_query.strip():
            from search_utils import semantic_search
            search_results = semantic_search(
                search_query, filtered_startups, top_k=50)
            filtered_startups = [startup for startup, score in search_results]

        # Ergebnis-Label schlank unter der Suchzeile – Clear-Button entfällt
        st.markdown(f"**Found&nbsp;{len(filtered_startups)}&nbsp;startup(s)**")

    # Grid-Rendering mit st.columns - direkt ohne extra Abstand
    for i in range(0, len(filtered_startups), 3):
        cols = st.columns(3, gap="large")
        row_startups = filtered_startups[i:i+3]

        for col_index, startup in enumerate(row_startups):
            if col_index < len(cols):
                with cols[col_index]:
                    with st.container(border=True):
                        st.subheader(startup.get('name', 'N/A'))
                        st.caption(startup.get('industry_focus', 'N/A'))

                        if st.button("🔍 View Dossier", key=f"dossier_{startup.get('id')}", use_container_width=True):
                            st.session_state.selected_startup = startup
                            st.rerun()


def main():
    """ Main application entry point """
    st.set_page_config(page_title="🚀 AI Startup Dossier",
                       page_icon="💡", layout="wide")

    startups = load_startup_data().get('startup_blueprints', [])

    # Routing
    if "selected_startup" in st.session_state and st.session_state.selected_startup is not None:
        selected_startup = st.session_state.selected_startup
        if st.button("← Back to Explorer"):
            st.session_state.selected_startup = None
            st.rerun()
        dossier_view(selected_startup, startups)
    else:
        display_startup_grid(startups)


if __name__ == "__main__":
    main()
