import os
from dotenv import load_dotenv
load_dotenv()

# tag::llm[]
# Create the LLM
from langchain_openai import ChatOpenAI
from langchain_nomic import NomicEmbeddings

#using openAI 
llm = ChatOpenAI(
    openai_api_key=os.getenv('OPENAI_API_KEY'),
    model_name="gpt-4o"
)

# using LM Studio LLMs
# llm = ChatOpenAI(
#     base_url=os.getenv('LMSTUDIO_BASE_URL'),
#     model_name="deepseek-r1-distill-llama-8b",
#     openai_api_key="lm-studio",
# )
# end::llm[]

# tag::embedding[]
# Create the Embedding model
from langchain_openai import OpenAIEmbeddings

# using openAI API KEY 
embeddings = OpenAIEmbeddings(
    openai_api_key=os.getenv('OPENAI_API_KEY'),
    model="text-embedding-3-large"
)

# using Nomic
# embedding_provider = NomicEmbeddings(
#     model="nomic-embed-text-v1.5",
# )
# end::embedding[]
