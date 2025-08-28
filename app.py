# import packages
import streamlit as st
import pandas as pd
import re
import os
import altair as alt
import openai

# ✅ Load OpenAI API key from Streamlit secrets
openai.api_key = st.secrets["OPENAI_API_KEY"]

client = openai.OpenAI()

@st.cache_data
def get_response(user_prompt, temperature):
    response = client.responses.create(
        model="gpt-4o",  # Use the latest chat model
        input=[
            {"role": "user", "content": user_prompt}  # Prompt
        ],
        temperature=temperature,  # Creativity control
        max_output_tokens=100     # Limit response length
    )
    return response

# Add a text input box for the user prompt
user_prompt = st.text_input("Enter your prompt:", "Explain generative AI in one sentence.")

# Add a slider for temperature
temperature = st.slider(
    "Model temperature:",
    min_value=0.0,
    max_value=1.0,
    value=0.7,
    step=0.01,
    help="Controls randomness: 0 = deterministic, 1 = very creative"
)

with st.spinner("AI is working..."):
    response = get_response(user_prompt, temperature)
    # print the response from OpenAI
    st.write(response.output[0].content[0].text)

def get_dataset_path():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(current_dir, "data", "customer_reviews.csv")
    return csv_path

def clean_text(text):
    text = text.lower().strip()
    text = re.sub(r'[^\w\s]', '', text)
    return text

# Layout two buttons side by side
col1, col2 = st.columns(2)

with col1:
    if st.button("📑 Ingest Dataset"):
        try:
            csv_path = get_dataset_path()
            st.session_state["df"] = pd.read_csv(csv_path)
            st.success("Dataset loaded successfully!")
        except FileNotFoundError:
            st.error("Dataset not found. Please check the file path.")

with col2:
    if st.button("🧼 Parse Reviews"):
        if "df" in st.session_state:
            st.session_state["df"]["CLEANED_SUMMARY"] = st.session_state["df"]["SUMMARY"].apply(clean_text)
            st.success("Reviews parsed and cleaned!")
        else:
            st.warning("Please ingest the dataset first.")

# Display the dataset if it exists
if "df" in st.session_state:
    # Product filter dropdown
    st.subheader("🔎 Filter by Product")
    product = st.selectbox("Choose a product", ["All Products"] + list(st.session_state["df"]["PRODUCT"].unique()))
    st.subheader(f"📁 Reviews for {product}")

    if product != "All Products":
        filtered_df = st.session_state["df"][st.session_state["df"]["PRODUCT"] == product]
    else:
        filtered_df = st.session_state["df"]
    st.dataframe(filtered_df)
    
    st.subheader(f"Sentiment Score Distribution for {product}")
    # Create Altair histogram using add_params instead of add_selection
    interval = alt.selection_interva_
