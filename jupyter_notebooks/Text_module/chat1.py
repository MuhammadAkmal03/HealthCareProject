# # ingest.py
# from langchain_pinecone import PineconeVectorStore
# from langchain.document_loaders import PyPDFLoader
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain.embeddings import HuggingFaceEmbeddings
# # from pinecone.grpc import PineconeGRPC as Pinecone
# from pinecone import Pinecone, ServerlessSpec

# from pinecone import ServerlessSpec
# import os


# from dotenv import load_dotenv
# import os

# # load environment variables from .env file
# load_dotenv()

# # now you can access them
# print("Pinecone API Key loaded?", os.getenv("PINECONE_API_KEY") is not None)


# # init Pinecone
# pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

# index_name = "medicalbotdata"

# # create index (only run once)
# if index_name not in pc.list_indexes().names():
#     pc.create_index(
#         name=index_name,
#         dimension=384,
#         metric="cosine",
#         spec=ServerlessSpec(cloud="aws", region="us-east-1"),
#     )

# # load PDF
# loader = PyPDFLoader("medical_knowledge.pdf")
# documents = loader.load()

# # split
# text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
# chunks = text_splitter.split_documents(documents)

# # embeddings
# embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# # push to Pinecone
# PineconeVectorStore.from_documents(chunks, embeddings, index_name=index_name)
# print(" Data ingested into Pinecone!")

from langchain_pinecone import PineconeVectorStore
from langchain.document_loaders import DirectoryLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from pinecone import Pinecone, ServerlessSpec
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()
print("Pinecone API Key loaded?", os.getenv("PINECONE_API_KEY") is not None)

# Init Pinecone
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

index_name = "medicalbotdata"

# Create index (only run once)
if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=384,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
    )

#  Load all PDFs from "documents" folder
loader = DirectoryLoader("documents", glob="*.pdf", loader_cls=PyPDFLoader)
documents = loader.load()

# Split into chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
chunks = text_splitter.split_documents(documents)

# Embeddings
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Push to Pinecone
PineconeVectorStore.from_documents(chunks, embeddings, index_name=index_name)
print("All PDFs ingested into Pinecone!")
