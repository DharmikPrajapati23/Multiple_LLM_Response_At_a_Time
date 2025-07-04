import streamlit as st
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnableParallel
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# --- Configuration ---
# Hugging Face API key from environment variables
HUGGINGFACE_API_KEY = os.getenv("HUGGINGFACE_API_KEY")

# Model IDs for different LLMs
LLAMA_ID = "meta-llama/Llama-3.3-70B-Instruct"
MISTRAL_ID = "mistralai/Mixtral-8x7B-Instruct-v0.1"
DEEPSEEK_ID = "deepseek-ai/DeepSeek-R1"
QWEN_ID = "Qwen/Qwen2.5-72B-Instruct"


# Check if API key is available
if not HUGGINGFACE_API_KEY:
    st.error("HUGGINGFACE_API_KEY not found in environment variables. Please set it up in a .env file.")
    st.stop()

# --- Langchain Setup ---
# Prompt for the question to be answered
prompt = PromptTemplate(
    template="""
    You are a helpful assistant. \n
    You will be given a question and you need to answer it based on the type of question.\n
    Type: {type}\n
    Question: {question}\n
    Answer the question in a concise manner.
    """,
    input_variables=["type", "question"]
)

# Initialize LLMs
@st.cache_resource
def initialize_llms():
    try:
        # LLAMA 3
        llama_llm = HuggingFaceEndpoint(
            repo_id=LLAMA_ID,
            huggingfacehub_api_token=HUGGINGFACE_API_KEY,
            task="conversational",
            temperature=0.1,
            max_new_tokens=512
        )
        llama_model = ChatHuggingFace(llm=llama_llm)
    except Exception as e:
        st.warning(f"Could not initialize Llama 3: {e}. Please check model ID and API key access.")
        llama_model = None

    try:
        # MISTRAL
        mistral_llm = HuggingFaceEndpoint(
            repo_id=MISTRAL_ID,
            huggingfacehub_api_token=HUGGINGFACE_API_KEY,
            task="conversational",
            temperature=0.1,
            max_new_tokens=512
        )
        mistral_model = ChatHuggingFace(llm=mistral_llm)
    except Exception as e:
        st.warning(f"Could not initialize Mistral: {e}. Please check model ID and API key access.")
        mistral_model = None

    try:
        # QWEN
        qwen_llm = HuggingFaceEndpoint(
            repo_id=QWEN_ID,
            huggingfacehub_api_token=HUGGINGFACE_API_KEY,
            task="conversational",
            temperature=0.1,
            max_new_tokens=512
        )
        qwen_model = ChatHuggingFace(llm=qwen_llm)
    except Exception as e:
        st.warning(f"Could not initialize Qwen: {e}. Please check model ID and API key access.")
        qwen_model = None

    try:
        # DEEPSEEK
        deepseek_llm = HuggingFaceEndpoint(
            repo_id=DEEPSEEK_ID,
            huggingfacehub_api_token=HUGGINGFACE_API_KEY,
            task="conversational",
            temperature=0.1,
            max_new_tokens=512
        )
        deepseeker_model = ChatHuggingFace(llm=deepseek_llm)
    except Exception as e:
        st.warning(f"Could not initialize DeepSeek: {e}. Please check model ID and API key access.")
        deepseeker_model = None

    return llama_model, mistral_model, qwen_model, deepseeker_model

llama_model, mistral_model, qwen_model, deepseeker_model = initialize_llms()

parser = StrOutputParser()

# Create a RunnableParallel to process the question through all available models
chains = {}
if llama_model:
    chains["Llama 3"] = prompt | llama_model | parser
if mistral_model:
    chains["Mistral"] = prompt | mistral_model | parser
if qwen_model:
    chains["Qwen"] = prompt | qwen_model | parser
if deepseeker_model:
    chains["DeepSeek"] = prompt | deepseeker_model | parser

parallel_chain = RunnableParallel(chains)


# --- Streamlit UI ---
st.set_page_config(page_title="LLM Question Answering", layout="wide")
st.title("🧠 LLM-Powered Question Answering for Exam Prep")
st.markdown("Ask a question and get answers from multiple open-source LLMs!")

# Input fields
question_types = ["Single-Select","Multi-Select", "True/False", "Short Answer", "Long Answer"]
selected_type = st.radio("Select Question Type:", question_types, horizontal=True)

user_question = st.text_area("Enter your question :", height=300)

# Submit button
if st.button("Get Answers from LLMs", type="primary"):
    if not user_question:
        st.warning("Please enter a question.")
    elif not chains:
        st.error("No LLM models could be initialized. Please check your API key and model access.")
    else:
        st.subheader("Generating Answers...")
        with st.spinner("Please wait while LLMs process your question..."):
            try:
                # Invoke the parallel chain
                results = parallel_chain.invoke({"type": selected_type, "question": user_question})

                st.success("Answers Generated!")
                st.markdown("---")

                # Display results in columns
                cols = st.columns(len(results))
                for i, (model_name, answer) in enumerate(results.items()):
                    with cols[i]:
                        st.markdown(f"### {model_name} Response")
                        st.info(answer)
                        st.markdown("---")

            except Exception as e:
                st.error(f"An error occurred during LLM invocation: {e}")
                st.info("Please ensure your Hugging Face API key is valid and the models are accessible.")

st.markdown("---")
