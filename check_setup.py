import faiss
print("faiss OK - version", faiss.__version__)

from langchain_community.vectorstores import FAISS
print("langchain_community FAISS OK")

from langchain_community.embeddings import HuggingFaceEmbeddings
print("HuggingFaceEmbeddings (community) OK")

from mistralai import Mistral
print("mistralai.Mistral OK")

from langchain_mistralai import ChatMistralAI, MistralAIEmbeddings
print("langchain_mistralai ChatMistralAI / MistralAIEmbeddings OK")