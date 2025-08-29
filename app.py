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

# ---------------- Tab 1: Chatbot (OpenAI Dependent + AI Sentiment) ----------------
with tab1:
    st.header("💬 LibChat - Your Expert Librarian")
    st.caption("On rush? Can't find a book? Ask LibChat!")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Render chat history
    for msg in st.session_state.messages:
        role = "assistant" if msg["role"] == "bot" else "user"
        avatar = "📚" if role == "assistant" else "🙂"
        with st.chat_message(role, avatar=avatar):
            st.markdown(msg["content"])

    # Chat input
    if user_input := st.chat_input("Type your message..."):
        st.session_state.messages.append({"role": "user", "content": user_input})

        # OpenAI response
        bot_reply = get_response(user_input)
        st.session_state.messages.append({"role": "bot", "content": bot_reply})

        # Rerun to show new message immediately
        st.rerun()

    # ---------------- Sentiment Analysis via OpenAI ----------------
    if st.session_state.messages:
        user_texts = " ".join([m["content"] for m in st.session_state.messages if m["role"] == "user"])
        if user_texts.strip():
            try:
                sentiment_response = openai.chat.completions.create(
                    model="gpt-4o",
                    messages=[
                        {"role": "system", "content": "You are a sentiment analysis assistant."},
                        {"role": "user", "content": f"Analyze the sentiment of the following messages. "
                                                    f"Categorize each as Positive, Neutral, or Negative, "
                                                    f"and give an overall polarity score (-1 to 1):\n{user_texts}"}
                    ],
                    temperature=0
                )

                sentiment_text = sentiment_response.choices[0].message.content.strip()
                st.markdown("---")
                st.subheader("📊 AI-Powered Sentiment Analysis")
                st.write(sentiment_text)

            except Exception as e:
                st.error(f"Failed to analyze sentiment: {e}")


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

    # ---------------- Chat Sentiment Trend ----------------
    st.subheader("Chat Sentiment Trend")
    if "messages" in st.session_state and st.session_state.messages:
        if "sentiment_score" in st.session_state:
            sentiment_scores = [st.session_state.sentiment_score] * len(
                [m for m in st.session_state.messages if m["role"]=="user"]
            )
        else:
            user_queries = [m["content"] for m in st.session_state.messages if m["role"]=="user"]
            sentiment_scores = [TextBlob(q).sentiment.polarity for q in user_queries]
        st.line_chart(sentiment_scores)

    # ---------------- Top User Topics (AI + Co-occurrence) ----------------
    st.subheader("Top User Topics")
    if "messages" in st.session_state and st.session_state.messages:
        user_texts = "\n".join([m["content"] for m in st.session_state.messages if m["role"]=="user"])
        if user_texts.strip():
            try:
                topic_response = openai.chat.completions.create(
                    model="gpt-4o",
                    messages=[
                        {"role": "system", "content": "You are an AI analyst for library chatbot interactions."},
                        {"role": "user", "content": f"Analyze these user messages and provide the top 5 most frequent topics. "
                                                    f"Also, list any topics that often appear together (co-occurrence).\n{user_texts}"}
                    ],
                    temperature=0
                )
                topic_summary = topic_response.choices[0].message.content.strip()
                st.markdown(topic_summary)
            except Exception as e:
                st.error(f"Failed to extract topics: {e}")

    # ---------------- Missed Search Analysis (AI) ----------------
    st.subheader("Missed Search Analysis")
    if "messages" in st.session_state:
        user_queries = [m["content"] for m in st.session_state.messages if m["role"]=="user"]
        missed_queries = [q for q in user_queries if keyword_search(q).empty]
        total_queries = len(user_queries)

        if total_queries > 0:
            st.write(f"Missed search queries: {len(missed_queries)} / {total_queries} ({len(missed_queries)/total_queries*100:.1f}%)")
            
            if missed_queries:
                try:
                    missed_response = openai.chat.completions.create(
                        model="gpt-4o",
                        messages=[
                            {"role": "system", "content": "You are an AI analyst for library catalog searches."},
                            {"role": "user", "content": f"These search queries returned no results: {missed_queries}\n"
                                                        "Analyze why users might not find results and suggest improvements."}
                        ],
                        temperature=0
                    )
                    missed_summary = missed_response.choices[0].message.content.strip()
                    st.markdown(missed_summary)
                except Exception as e:
                    st.error(f"Failed to analyze missed searches: {e}")

    # ---------------- Most Searched Books ----------------
    st.subheader("Most Searched Books")
    if "search_history" not in st.session_state:
        st.session_state.search_history = []

    # Append latest search from Tab 2 if exists
    if search_query:
        st.session_state.search_history.append(search_query)

    if st.session_state.search_history:
        most_searched = pd.Series(st.session_state.search_history).value_counts().head(10)
        st.bar_chart(most_searched)




