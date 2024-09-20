import os
import PyPDF2
import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import PromptTemplate
from langchain_openai import AzureOpenAIEmbeddings, AzureChatOpenAI
from dotenv import load_dotenv
from io import BytesIO

# Load environment variables
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
    return all_text

# Streamlit app
def main():
    st.title("PDF Summarizer")

    # Upload a PDF
    uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")

    if uploaded_file:
        # Process the PDF
        with st.spinner('Processing ...'):
            pdf_path = "temp_uploaded_pdf.pdf"
            with open(pdf_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            # Extract text from the PDF
            all_text = extract_text_from_pdf(pdf_path)

            # Split text into chunks
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=2000,
                chunk_overlap=300,
                length_function=len,
                separators=['\n', '\n\n', ' ', '']
            )
            chunks = text_splitter.split_text(text=all_text)

            # Set up LLM for summarization
            llm = AzureChatOpenAI(deployment_name="gpt-4o-mini")
            
            # Define prompt for chunk summarization
            summary_prompt_template = """
            Summarize the following text in clear and concise terms:
            {chunk}
            """
            prompt = PromptTemplate.from_template(template=summary_prompt_template)

            # Summarize each chunk
            chunk_summaries = []
            for chunk in chunks:
                formatted_prompt = prompt.format(chunk=chunk)
                chunk_summary = llm.invoke(formatted_prompt)  # Use LLM's predict method
                chunk_summaries.append(chunk_summary.content)

            # Combine chunk summaries into a final summary
            st.subheader("Final Summary of the Document")
            final_summary = "\n".join(chunk_summaries)
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
