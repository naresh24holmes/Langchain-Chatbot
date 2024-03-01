from langchain_community.llms import HuggingFaceHub
import streamlit as st
import os
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("HUGGINGFACEHUB_API_TOKEN")

def get_huggingface_response(question):
    # Instantiate HuggingFaceHub with the API token directly
    llm = HuggingFaceHub(
        huggingfacehub_api_token="api_key",
        repo_id="google/flan-t5-large",
        model_kwargs={"temperature": 0.6, "max_length": 64}
    )

    # Get response from the model
    response = llm(question)

    return response


st.set_page_config(page_title="Q&A Chatbot using Gemma")
st.header("Coversational Chatbot")


input_text = st.text_input("Input:", key="input")

submit_button = st.button("Ask the question")

# If the submit button is clicked
if submit_button:
 
    response = get_huggingface_response(input_text)
 
    st.subheader("The response is:")
    st.write(response)
