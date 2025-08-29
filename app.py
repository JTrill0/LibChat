import streamlit as st
import pandas as pd
import altair as alt
from textblob import TextBlob
from dotenv import load_dotenv
import openai
import os
import re
from rapidfuzz import fuzz  # ✅ Better fuzzy matching

# ---- Custom CSS for Chat History ----
st.markdown(
    """
    <style>
    /* Chat area background */
    div[data-testid="stChatMessageContainer"] {
        background-color: #f0f2f6;
        padding: 15px;
        border-radius: 12px;
    }

    /* Base bubble styling */
    div[data-testid="stChatMessage"] {
        max-width: 70%;
        padding: 10px 15px;
        margin: 5px 0;
        border-radius: 18px;
        word-wrap: break-word;
    }

    /* User bubble (right, blue) */
    div[data-testid="stChatMessage"][data-testid="user"] > div {
        background-color: #ffffff !important;
        color: white !important;
        margin-left: auto;   /* push to right */
        text-align: left;
        border-bottom-right-radius: 4px;
    }

    /* Assistant bubble (left, gray) */
    div[data-testid="stChatMessage"][data-testid="assistant"] > div {
        background-color: #e5e5ea !important;
        color: black !important;
        margin-right: auto;  /* push to left */
        text-align: left;
        border-bottom-left-radius: 4px;
    }
    </style>
    """,
    unsafe_allow_html=True
)


# ---------------- Load Environment ----------------
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# ---------------- Load Books Database ----------------
@st.cache_data
def load_books():
    file_path = os.path.join("data", "books_1000.xlsx")
    if not os.path.exists(file_path):
        st.error(f"❌ Could not find books database at: {os.path.abspath(file_path)}")
        return pd.DataFrame()

    df = pd.read_excel(file_path)

    expected_cols = ["ISBN", "TITLE", "AUTHOR", "DATE PUBLISH", "GENRE", "SECTION", "QUANTITY", "TOPICS", "OVERVIEW"]
    for col in expected_cols:
        if col not in df.columns:
            df[col] = ""
    return df

books_df = load_books()

# ---------------- Intent Detection ----------------
def is_book_query(user_input: str) -> bool:
    book_keywords = [
        "book", "novel", "read", "catalog", "library", "title", "isbn",
        "related to", "overview", "author", "published", "project"
    ]
    text = user_input.lower()
    return any(keyword in text for keyword in book_keywords)

# ---------------- Keyword Search (Catalog) ----------------
def keyword_search(query, df=None):
    if df is None:
        df = books_df
    if df.empty or not query:
        return pd.DataFrame()

    query = re.sub(r"[^a-z0-9\s]", "", query.lower())
    keywords = query.split()
    mask = pd.Series(False, index=df.index)
    for kw in keywords:
        mask |= df["TITLE"].astype(str).str.lower().str.contains(kw, na=False)
        mask |= df["AUTHOR"].astype(str).str.lower().str.contains(kw, na=False)
        mask |= df["TOPICS"].astype(str).str.lower().str.contains(kw, na=False)
        mask |= df["OVERVIEW"].astype(str).str.lower().str.contains(kw, na=False)
    return df[mask].head(10)

# ---------------- Fuzzy Search (Chat) ----------------
def fuzzy_search(query, df=None, threshold=75):
    if df is None:
        df = books_df
    if df.empty or not query:
        return pd.DataFrame()

    results = []
    for _, row in df.iterrows():
        combined_text = " ".join(str(row[col]) for col in ["TITLE", "AUTHOR", "TOPICS", "OVERVIEW"])
        score = fuzz.partial_ratio(query.lower(), combined_text.lower())
        if score >= threshold:
            results.append((score, row))

    if not results:
        return pd.DataFrame()

    results = sorted(results, key=lambda x: x[0], reverse=True)
    return pd.DataFrame([r[1] for r in results[:10]])

# ---------------- OpenAI Response ----------------
@st.cache_data(show_spinner=False)
def get_response(user_prompt, temperature=0.5):
    response = openai.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are LibChat, an expert librarian who helps students find books and provides knowledge support."},
            {"role": "user", "content": user_prompt}
        ],
        temperature=temperature,
        max_tokens=150
    )
    return response.choices[0].message.content.strip()

# ---------------- UI ----------------
st.title("📚 LibChat System")
st.write("Your AI-Powered Librarian and Library Catalog! 🔎")

tab1, tab2, tab3 = st.tabs(["💬 Chatbot", "📖 Library Catalog", "📊 Data Analytics"])

with tab1:
    st.header("💬 LibChat - Your Expert Librarian")
    st.caption("On rush? Can't find a book? Ask LibChat!")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Render all previous messages
    for msg in st.session_state.messages:
        if msg["role"] == "bot":
            with st.chat_message("assistant", avatar="📚"):
                st.markdown(msg["content"])
        else:
            with st.chat_message("user", avatar="🙂"):
                st.markdown(msg["content"])

    # Chat input
    if user_input := st.chat_input("Type your message..."):
        st.session_state.messages.append({"role": "user", "content": user_input})

        # Fully OpenAI dependent response
        bot_reply = get_response(user_input)
        st.session_state.messages.append({"role": "bot", "content": bot_reply})

        # Rerun to show new message above input
        st.rerun()

    # Sentiment analysis
    if st.session_state.messages:
        user_texts = " ".join([m["content"] for m in st.session_state.messages if m["role"] == "user"])
        if user_texts.strip():
            analysis = TextBlob(user_texts)
            polarity = analysis.sentiment.polarity
            sentiment = "😊 Positive" if polarity > 0.1 else "😞 Negative" if polarity < -0.1 else "😐 Neutral"
            st.markdown("---")
            st.subheader("📊 Sentiment Analysis (Conversation History)")
            st.write(f"**Overall Sentiment:** {sentiment}")
            st.write(f"**Polarity Score:** {polarity:.2f}")
 
# ---------------- Tab 2: Library Catalog ----------------
with tab2:
    st.header("📖 Explore the Library Collection")
    search_query = st.text_input("🔍 Search books by title, author, topics, or overview")
    filtered_books = books_df.copy()
    if search_query:
        filtered_books = keyword_search(search_query, df=filtered_books)

    # Filters
    col1, col2, col3 = st.columns(3)
    with col1:
        genre_filter = st.multiselect("Filter by Genre", options=sorted(books_df["GENRE"].dropna().unique()))
    with col2:
        section_filter = st.multiselect("Filter by Section", options=sorted(books_df["SECTION"].dropna().unique()))
    with col3:
        year_filter = st.multiselect("Filter by Year", options=sorted(books_df["DATE PUBLISH"].dropna().unique()))

    if genre_filter:
        filtered_books = filtered_books[filtered_books["GENRE"].isin(genre_filter)]
    if section_filter:
        filtered_books = filtered_books[filtered_books["SECTION"].isin(section_filter)]
    if year_filter:
        filtered_books = filtered_books[filtered_books["DATE PUBLISH"].isin(year_filter)]

    st.subheader("📑 Search Results")
    if not filtered_books.empty:
        st.dataframe(filtered_books[["ISBN", "TITLE", "AUTHOR", "DATE PUBLISH", "GENRE", "SECTION", "QUANTITY", "TOPICS", "OVERVIEW"]])
    else:
        st.info("No books found. Try adjusting your search or filters.")


# ---------------- Tab 3: Data Analytics ----------------
with tab3:
    st.header("📊 Library Data Analytics")
    st.write("Insights from library catalog and chatbot interactions")

    # Top Genres & Authors
    st.subheader("Top Genres")
    st.bar_chart(books_df["GENRE"].value_counts().head(10))
    st.subheader("Top Authors")
    st.bar_chart(books_df["AUTHOR"].value_counts().head(10))

    # Publication Trends
    st.subheader("Books Published Over the Years by Genre")
    trend_chart = alt.Chart(books_df).mark_bar().encode(
        x="DATE PUBLISH:O",
        y="count()",
        color="GENRE:N",
        tooltip=["GENRE", "DATE PUBLISH", "count()"]
    ).properties(title="Books Published per Year by Genre")
    st.altair_chart(trend_chart, use_container_width=True)

    # Chat Sentiment Trend
    st.subheader("Chat Sentiment Trend")
    if "messages" in st.session_state and st.session_state.messages:
        # Use OpenAI sentiment if available, else fallback
        if "sentiment_score" in st.session_state:
            sentiment_scores = [st.session_state.sentiment_score] * len(
                [m for m in st.session_state.messages if m["role"]=="user"]
            )
        else:
            # Fallback to TextBlob
            user_queries = [m["content"] for m in st.session_state.messages if m["role"]=="user"]
            sentiment_scores = [TextBlob(q).sentiment.polarity for q in user_queries]

        st.line_chart(sentiment_scores)

    # Missed Searches
    st.subheader("Missed Search Queries")
    if "messages" in st.session_state:
        user_queries = [m["content"] for m in st.session_state.messages if m["role"]=="user"]
        missed = sum(1 for q in user_queries if keyword_search(q).empty)
        total = len(user_queries)
        if total > 0:
            st.write(f"Missed search queries: {missed} / {total} ({missed/total*100:.1f}%)")




