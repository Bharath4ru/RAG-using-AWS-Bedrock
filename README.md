# Chat with PDF using AWS Bedrock

This project allows you to interact with PDF files by asking questions, with responses generated using AWS Bedrock's Titan Embeddings and Llama models. The application employs Streamlit for the user interface, enabling dynamic PDF uploads, vector store creation, and LLM-powered Q&A.

---

## Features

- **PDF Upload**: Upload multiple PDF files directly through the application.
- **Data Ingestion**: Extracts and splits text from uploaded PDF files for efficient processing.
- **Vector Store**: Creates a vector store using FAISS and Titan Embeddings for document retrieval.
- **Q&A Interface**: Ask questions based on the uploaded PDFs and receive concise, detailed answers.
- **AWS Bedrock Integration**: Leverages Amazon's Bedrock for embeddings and LLM capabilities.

---

## Requirements

### Python Packages:
- `boto3`
- `streamlit`
- `langchain`

Install these dependencies using pip:
```bash
pip install boto3 streamlit langchain
```

### AWS Configuration:
To use AWS Bedrock, configure your AWS credentials:

1. Run the following command to configure AWS CLI:
   ```bash
   aws configure
   ```

2. Provide the required details:
   - **Access Key ID**: Your AWS Access Key ID
   - **Secret Access Key**: Your AWS Secret Access Key
   - **Default region**: `ap-south-1`

Ensure that your AWS credentials have the necessary permissions to access Bedrock services.

---

## How to Run

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd <repository-name>
   ```

2. Run the Streamlit app:
   ```bash
   streamlit run app.py
   ```

3. Open the application in your browser at `http://localhost:8501`.

---

## How It Works

### 1. **Data Ingestion**
Uploaded PDF files are processed by:
- Saving the files temporarily.
- Using `PyPDFLoader` to extract text.
- Splitting text into chunks for embedding using `RecursiveCharacterTextSplitter`.

### 2. **Vector Store Creation**
- Titan Embeddings generate vector embeddings for the document chunks.
- FAISS stores these embeddings for efficient similarity searches.

### 3. **Question Answering**
- The uploaded PDF data is retrieved using FAISS.
- A prompt template structures the input to the Llama model.
- AWS Bedrock generates answers based on the retrieved context.

---

## Application Workflow

1. **Upload PDFs**: Users can upload one or more PDF files via the sidebar.
2. **Update Vectors**: After uploading, click `Vectors Update` to process and store the PDF data in the vector store.
3. **Ask Questions**: Enter a question in the main interface to receive answers based on the uploaded documents.

---

## Future Enhancements

- Add support for additional file formats.
- Optimize the vector store creation for large document sets.
- Implement authentication for secure access.

---

## License

This project is licensed under the [MIT License](LICENSE).

---

## Acknowledgments

- **Streamlit**: For the interactive web application framework.
- **AWS Bedrock**: For providing advanced LLM and embedding services.
- **LangChain**: For the tools to manage embeddings, vector stores, and LLMs.
