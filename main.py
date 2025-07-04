from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnableSequence, RunnableParallel

import os
from dotenv import load_dotenv
load_dotenv()


# Hugging Face API key from environment variables`
HUGGINGFACE_API_KEY = os.getenv("HUGGINGFACE_API_KEY")

# Model IDs for different LLMs
LLAMA_ID = "meta-llama/Llama-3.3-70B-Instruct"
MISTRAL_ID = "mistralai/Mixtral-8x7B-Instruct-v0.1"
DEEPSEEK_ID = "deepseek-ai/DeepSeek-R1"
QWEN_ID = "Qwen/Qwen2.5-72B-Instruct"



# Prompt for the question to be answered
prompt = PromptTemplate(
    template="""
    You are a helpful assistant. \n
    You will be given a question and you need to answer it based on the type of question.\n
    Type: {type}\n
    Question: {question}\n
    Answer the question in a concise manner
""",
    input_variables=["type","question"]
)

#------------------LLAMA 3.3------------------#
llama_llm = HuggingFaceEndpoint(
    repo_id=LLAMA_ID,
    huggingfacehub_api_token=HUGGINGFACE_API_KEY,
    task="conversational"
)

llama_model = ChatHuggingFace(llm=llama_llm)

#------------------MISTRAL------------------#

mistral_llm = HuggingFaceEndpoint(
    repo_id=MISTRAL_ID,
    huggingfacehub_api_token=HUGGINGFACE_API_KEY,
    task="conversational"
)

mistral_model = ChatHuggingFace(llm=mistral_llm)



#------------------QWEN------------------#

qwen_llm = HuggingFaceEndpoint(
    repo_id=QWEN_ID,
    huggingfacehub_api_token=HUGGINGFACE_API_KEY,
    task="conversational"
)

qwen_model = ChatHuggingFace(llm=qwen_llm)


#------------------DEEPSEEK------------------#

deepseek_llm = HuggingFaceEndpoint(
    repo_id=DEEPSEEK_ID,
    huggingfacehub_api_token=HUGGINGFACE_API_KEY,
    task="conversational"
)

deepseeker_model = ChatHuggingFace(llm=deepseek_llm)


#------------------------------------------------------------


parser = StrOutputParser()



# Create a RunnableSequence to process the question through all models

parallel_chain = RunnableParallel({
    "llama": prompt | llama_model | parser,
    "mistral": prompt | mistral_model | parser,
    "qwen": prompt | qwen_model | parser,
    "deepseek": prompt | deepseeker_model | parser
})

type = "True/False" # Multi-Select, Single-Select, True/False, Short Answer, Long Answer

question = """
MongoDB supports ACID transactions.
"""

result = parallel_chain.invoke({"type": type, "question": question})

print("LLAMA RESULT:", result["llama"])
print("MISTRAL RESULT:", result["mistral"])
print("QWEN RESULT:", result["qwen"])
print("DEEPSEEK RESULT:", result["deepseek"])