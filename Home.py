import logging
import os
import streamlit as st
from langchain.agents import create_csv_agent
from langchain.llms import OpenAI
import tempfile
import torch
import pandas as pd

# Check if GPU is available and set the device accordingly
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    st.set_page_config(page_title="Ask your CSV")
    st.title("Talk to your CSV ")

    # Input field for OpenAI API key
    openai_api_key = st.text_input(
        "Enter your OpenAI API key to proceed:", type='password')
    if openai_api_key:
        os.environ["OPENAI_API_KEY"] = openai_api_key

        # Check if the OpenAI API key is set
        if os.getenv("OPENAI_API_KEY") is None or os.getenv("OPENAI_API_KEY") == "":
            logger.error("OPENAI_API_KEY is not set")
            st.error(
                "API Key not set. Please set the OPENAI_API_KEY environment variable.")
            return

        csv_file = st.file_uploader("Upload a CSV file", type="csv")
        if csv_file is not None:
            try:
                # Save the uploaded file temporarily
                with tempfile.NamedTemporaryFile(delete=False) as temp_file:
                    temp_file.write(csv_file.read())
                    csv_file_path = temp_file.name
                    # Display the uploaded CSV data as a DataFrame
                    csv_data = pd.read_csv(csv_file_path)

                agent = create_csv_agent(
                    OpenAI(temperature=0), csv_file_path, verbose=True)

                user_question = st.text_input(
                    "Ask a question about your CSV: ")

                if user_question:
                    try:
                        with st.spinner(text="In progress..."):
                            result = agent.run(user_question)
                        st.info(result, icon="🤖")
                        st.dataframe(csv_data)

                    except Exception as e:
                        logger.error(f"An error occurred: {e}")
                        st.error(
                            f"An error occurred: {e}")

            finally:
                os.unlink(csv_file_path)  # Clean up the temporary file


if __name__ == "__main__":
    main()
