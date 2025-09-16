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
    llm_model = ChatGroq(model=llm)
    output_parser = StrOutputParser()
    chain = prompt | llm_model | output_parser
    answer = chain.invoke({"question": question})
    return answer
