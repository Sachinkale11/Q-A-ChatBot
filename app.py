import streamlit as st
import os
from langchain_groq import ChatGroq
import groq
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv

load_dotenv()

## langsmith tracking
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGSMITH_TRACING"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "Q&A_ChatBot"

## Prompt template

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant that helps people find information."),
    ("user", "Answer the following question: {question}")
])

def generate_response(question, api_key, llm, temperature, max_tokens):
    groq.api_key = api_key
    llm_model = ChatGroq(model=llm, temperature=temperature)
    output_parser = StrOutputParser()
    chain = prompt | llm_model | output_parser
    answer = chain.invoke({"question": question})
    return answer


## Title of the page
st.title("Q&A ChatBot")

## Sidebar for settings
st.sidebar.title("Settings")
api_key = st.sidebar.text_input("Enter your Groq API Key:", type="password")

## Dropdown for model selection
llm = st.sidebar.selectbox("Select a Groq AI Model", ["llama-3.1-8b-instant", "llama-3.3-70b-versatile", "openai/gpt-oss-120b", "openai/gpt-oss-20b"])

## Adjust response parameters
temperature = st.sidebar.slider("Temperature",min_value=0.0,max_value=1.0,value=0.7)
max_tokens = st.sidebar.slider("Max Tokens",min_value=50,max_value=300,value=150)

## Main Interface for user input
st.write("Ask any question")
user_question = st.text_input("Your Question:")

if user_question:
    if not api_key:
        st.error("Please enter your Groq API Key in the sidebar.")
    else:
        with st.spinner("Generating response..."):
            response = generate_response(user_question, api_key, llm, temperature, max_tokens)
        st.markdown("### Response:")
        st.write(response)
else:
    st.write("Please enter a question to get started.")
