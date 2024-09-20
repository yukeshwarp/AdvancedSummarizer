import os
import re
import numpy as np
import faiss
import PyPDF2
import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import PromptTemplate
from langchain_openai import AzureOpenAIEmbeddings, AzureChatOpenAI
from dotenv import load_dotenv
from io import BytesIO
from tqdm import tqdm

load_dotenv()

# Get Azure OpenAI environment variables from .env file
azure_api_key = os.getenv("AZURE_OPENAI_API_KEY")
azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
azure_api_type = os.getenv("OPENAI_API_TYPE")
azure_api_version = os.getenv("OPENAI_API_VERSION")

# Set the environment variables
os.environ["AZURE_OPENAI_API_KEY"] = azure_api_key
os.environ["AZURE_OPENAI_ENDPOINT"] = azure_endpoint
os.environ["OPENAI_API_TYPE"] = azure_api_type
os.environ["OPENAI_API_VERSION"] = azure_api_version

# Function to clean text
def clean_text(text):
    cleaned_text = re.sub(r' +', ' ', cleaned_text)
    cleaned_text = re.sub(r'[\x00-\x1F]', '', cleaned_text)
    cleaned_text = cleaned_text.replace('\n', ' ')
    cleaned_text = re.sub(r'\s*-\s*', '', cleaned_text)
    return cleaned_text

# Function to extract text from PDF
def extract_text_from_pdf(pdf_path):
    all_text = ""
    with open(pdf_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        number_of_pages = len(reader.pages)
        for page_num in range(number_of_pages):
            page = reader.pages[page_num]
            page_text = page.extract_text()
            all_text += page_text
    return all_text, number_of_pages

# Streamlit app
def main():
    st.title("PDF Summarizer with Clustering")

    # Upload a PDF
    uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")

    if uploaded_file:
        with st.spinner('Processing ...'):
            pdf_path = "temp_uploaded_pdf.pdf"
            with open(pdf_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            all_text, total_pages = extract_text_from_pdf(pdf_path)
            cleaned_text = clean_text(all_text)

            # Split text into chunks
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=2000,
                chunk_overlap=300,
                length_function=len,
                separators=['\n', '\n\n', ' ', '']
            )
            chunks = text_splitter.split_text(text=cleaned_text)

            # Initialize embeddings
            embeddings = AzureOpenAIEmbeddings(
                model="text-embedding-3-large",
                deployment="TextEmbeddingLarge",
                api_version=azure_api_version,
                azure_endpoint=azure_endpoint,
                openai_api_key=azure_api_key
            )

            # Create document embeddings
            doc_embeddings = [embeddings.embed(chunk) for chunk in chunks]
            array = np.array(doc_embeddings).astype('float32')

            # Clustering with FAISS
            num_clusters = 50
            dimension = array.shape[1]
            kmeans = faiss.Kmeans(dimension, num_clusters, niter=20, verbose=True)
            kmeans.train(array)

            # Create a new index for the original dataset
            index = faiss.IndexFlatL2(dimension)
            index.add(array)

            # Fetch centroids
            D, I = index.search(kmeans.centroids, 1)

            # Summarization chain
            llm = AzureChatOpenAI(deployment_name="gpt-4o-mini")
            summary_prompt_template = """
            Summarize the following text comprehensively:
            {chunk}
            """
            prompt = PromptTemplate.from_template(template=summary_prompt_template)

            final_summary = ""
            for idx in tqdm(I.flatten(), desc="Processing documents"):
                chunk = chunks[idx]
                formatted_prompt = prompt.format(chunk=chunk)
                chunk_summary = llm.invoke(formatted_prompt)
                final_summary += chunk_summary + "\n"

            st.subheader("Final Summary of the Document")
            st.write(final_summary)

            # Download summary as .txt
            def download_txt(summary):
                return BytesIO(summary.encode('utf-8'))

            st.download_button(
                label="Download Summary as .txt",
                data=download_txt(final_summary),
                file_name="summary.txt",
                mime="text/plain"
            )

if __name__ == "__main__":
    main()
