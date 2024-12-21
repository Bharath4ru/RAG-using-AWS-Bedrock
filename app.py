import json
import os
import tempfile
import boto3
import streamlit as st

# We will be using Titan Embeddings Model to generate Embeddings
from langchain_community.embeddings import BedrockEmbeddings
from langchain.llms.bedrock import Bedrock

# Data Ingestion
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

# Vector Embedding and Vector Store
from langchain.vectorstores import FAISS

# LLm Models
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA

# Bedrock Clients
bedrock = boto3.client(service_name="bedrock-runtime", region_name="ap-south-1")
bedrock_embeddings = BedrockEmbeddings(model_id="amazon.titan-embed-text-v2:0", client=bedrock)

def data_ingestion(uploaded_file):
    # Save the uploaded file to a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
        temp_file.write(uploaded_file.read())
        temp_file_path = temp_file.name

    try:
        loader = PyPDFLoader(temp_file_path)
        documents = loader.load()

        # - in our testing, Character split works better with this PDF dataset
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
        docs = text_splitter.split_documents(documents)
    finally:
        # Clean up the temporary file
        os.remove(temp_file_path)

    return docs

def get_vector_store(docs):
    vectorstore_faiss = FAISS.from_documents(
        docs,
        bedrock_embeddings
    )
    vectorstore_faiss.save_local("faiss_index")

def get_llama2_llm():
    # Create the Meta Llama Model
    llm = Bedrock(model_id="meta.llama3-70b-instruct-v1:0", client=bedrock,
                  model_kwargs={'max_gen_len': 512})
    return llm

prompt_template = """
Human: Use the following pieces of context to provide a 
concise answer to the question at the end but summarize with 
at least 100 words and detailed explanations. If you don't know the answer, 
just say that you don't know; don't try to make up an answer.
<context>
{context}
</context>

Question: {question}

Assistant:"""

PROMPT = PromptTemplate(
    template=prompt_template, input_variables=["context", "question"]
)

def get_response_llm(llm, vectorstore_faiss, query):
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore_faiss.as_retriever(
            search_type="similarity", search_kwargs={"k": 3}
        ),
        return_source_documents=True,
        chain_type_kwargs={"prompt": PROMPT}
    )
    answer = qa({"query": query})
    return answer['result']

def main():
    st.set_page_config("Chat PDF")

    st.header("Chat with PDF using AWS Bedrock💁")

    user_question = st.text_input("Ask a Question from the PDF Files")

    with st.sidebar:
        st.title("Update or Create Vector Store:")

        uploaded_files = st.file_uploader("Upload PDF Files", type="pdf", accept_multiple_files=True)
        
        if st.button("Vectors Update"):
            if uploaded_files:
                with st.spinner("Processing..."):
                    all_docs = []
                    for uploaded_file in uploaded_files:
                        docs = data_ingestion(uploaded_file)
                        all_docs.extend(docs)
                    get_vector_store(all_docs)
                    st.success("Vector store updated successfully!")
            else:
                st.warning("Please upload PDF files to update the vector store.")

    if st.button("Llama3 Output"):
        with st.spinner("Processing..."):
            # Load the FAISS index with dangerous deserialization enabled
            faiss_index = FAISS.load_local("faiss_index", bedrock_embeddings, allow_dangerous_deserialization=True)
            llm = get_llama2_llm()

            st.write(get_response_llm(llm, faiss_index, user_question))
            st.success("Done")

if __name__ == "__main__":
    main()
