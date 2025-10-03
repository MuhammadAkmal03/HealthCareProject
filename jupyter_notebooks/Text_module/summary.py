# Imports and Setup
import os
from dotenv import load_dotenv

# LangChain components
from langchain_community.document_loaders import PyPDFLoader
from langchain.chains.summarize import load_summarize_chain
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.docstore.document import Document
from langchain.prompts import PromptTemplate

# Load environment variables
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")

# Check API key
if not api_key:
    raise ValueError("Google API Key not found. Please set it in the .env file.")
else:
    print("Libraries imported and API key loaded successfully.")

# ---------------------------
# Initialize LLM
# ---------------------------
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash", google_api_key=api_key, temperature=0.5
)

# ---------------------------
# Prompt Template
# ---------------------------
summary_prompt_template = """
Write a concise summary of the following medical text. 
Focus on the key findings, results, and recommendations.
Address the user directly using "your" or "you" (e.g., "Your results show...").
Do NOT include any personal identifying information such as names or ages.

"{text}"

CONCISE SUMMARY:
"""
summary_prompt = PromptTemplate(
    template=summary_prompt_template, input_variables=["text"]
)

# Load summarization chain
summarize_chain = load_summarize_chain(
    llm=llm,
    chain_type="map_reduce",
    map_prompt=summary_prompt,
    combine_prompt=summary_prompt,
)


# ---------------------------
# Helper Functions for Data Extraction
# ---------------------------
def process_pdf(file_path: str) -> list:
    """Loads a PDF and returns its content as a list of LangChain Documents."""
    try:
        loader = PyPDFLoader(file_path)
        return loader.load()
    except Exception as e:
        print(f"Error processing PDF: {e}")
        return []


def process_text(raw_text: str) -> list:
    """Takes a raw text string and returns it as a Document."""
    return [Document(page_content=raw_text)]


# ---------------------------
# Summarization
# ---------------------------
# Choose the input type: 'pdf' or 'text'
input_type = "pdf"
input_data = "./med_report/test_report.pdf"

# Process the input to get a list of documents
docs_to_summarize = []
if input_type == "pdf":
    docs_to_summarize = process_pdf(input_data)
elif input_type == "text":
    docs_to_summarize = process_text(input_data)
else:
    print("Invalid input_type. Please choose 'pdf' or 'text'.")

# Run summarization if documents were processed
if docs_to_summarize:
    print(f"\nStarting summarization for '{input_type}' input...")

    # Run the chain on the processed documents
    summary_result = summarize_chain.invoke(docs_to_summarize)

    # Print the final summary
    print("\n" + "-" * 20 + " Summary " + "-" * 20)
    print(summary_result["output_text"])
else:
    print(
        "\nCould not proceed with summarization due to an error in the previous step."
    )
