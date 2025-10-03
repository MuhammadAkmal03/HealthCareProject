# # chat2.py
# from dotenv import load_dotenv
# import os
# from langchain_pinecone import PineconeVectorStore
# from langchain_community.embeddings import HuggingFaceEmbeddings
# from langchain_google_genai import ChatGoogleGenerativeAI
# from langchain.chains import RetrievalQA

# # load .env
# load_dotenv()

# index_name = "medicalbot"

# # embeddings (must match ingestion)
# embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# # connect to existing Pinecone index
# vectorstore = PineconeVectorStore.from_existing_index(index_name, embeddings)

# retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# llm = ChatGoogleGenerativeAI(
#     model="gemini-2.5-flash",
#     google_api_key=os.getenv("GOOGLE_API_KEY"),
#     temperature=0.3
# )

# qa_chain = RetrievalQA.from_chain_type(
#     llm=llm, retriever=retriever
# )

# # interactive chat loop
# while True:
#     query = input("\nYou: ")
#     if query.lower() in ["quit", "exit"]:
#         break
#     result = qa_chain.invoke({"query": query})
#     print("Bot:", result["result"])
# chat2.py
from dotenv import load_dotenv
import os

from langchain_pinecone import PineconeVectorStore
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate

# ---------------------------
# 1️⃣ Load environment variables
# ---------------------------
load_dotenv()

# ---------------------------
# 2️⃣ Connect to Pinecone
# ---------------------------
index_name = "medicalbot"  # must match your Pinecone index

# Embeddings (same as ingestion)
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Connect to existing Pinecone index
vectorstore = PineconeVectorStore.from_existing_index(index_name, embeddings)

retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# ---------------------------
# 3️⃣ Initialize LLM
# ---------------------------
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=os.getenv("GOOGLE_API_KEY"),
    temperature=0.3,
)

# ---------------------------
# 4️⃣ Conversation Memory
# ---------------------------
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# ---------------------------
# 5️⃣ Prompt Template
# ---------------------------
prompt_template = """
You are a helpful medical assistant chatbot. Use the provided context to answer the question.
If the question is a follow-up, consider the previous conversation.

Chat History:
{chat_history}

Context from documents:
{context}

Question:
{question}

Answer concisely and accurately.
"""

PROMPT = PromptTemplate(
    template=prompt_template, input_variables=["chat_history", "context", "question"]
)

# ---------------------------
# 6️⃣ Conversational Retrieval Chain
# ---------------------------
qa_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=retriever,
    memory=memory,
    combine_docs_chain_kwargs={"prompt": PROMPT},
)

# ---------------------------
# 7️⃣ Interactive Chat Loop
# ---------------------------
print("✅ Chatbot ready! Type 'exit' or 'quit' to stop.")

while True:
    query = input("\nYou: ")
    if query.lower() in ["quit", "exit"]:
        print("Goodbye!")
        break

    result = qa_chain({"question": query})
    print("Bot:", result["answer"])
