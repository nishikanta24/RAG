from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings  #for vector embeddings of chunks
from langchain_community.vectorstores import FAISS #FAISS (Facebook AI Similarity Search) is a fast and efficient library for storing and searching vector embeddings.
                                                   #This import allows you to store and retrieve embeddings for similarity search in LangChain.

# Step 1 : Load raw pdf
DATA_PATH="data/"
def load_pdf_files(data):
    loader = DirectoryLoader(data,glob='*.pdf',loader_cls=PyPDFLoader)

    documents=loader.load()
    return documents

documents=load_pdf_files(data=DATA_PATH)
#print("Length of PDF pages: ",len(documents))


# Step 2 : Create chunks

def create_chunks(extracted_data):
    text_splitter=RecursiveCharacterTextSplitter(chunk_size=500,chunk_overlap=50)## chunk_sizre and chunk_overlap can be altered 
    text_chunks=text_splitter.split_documents(extracted_data)
    return text_chunks

text_chunks=create_chunks(extracted_data=documents)
#print("Length of text chunks: ",len(text_chunks))


# Step 3 : Create Vector Embeddings
def get_embedding_model():
    embedding_model=HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return embedding_model

embedding_model=get_embedding_model()

'''sentence-transformers/all-MiniLM-L6-v2 is a pretrained embedding model from the Sentence-Transformers library.
It is commonly used for converting text into numerical embeddings for NLP tasks like:

✅ Semantic Search
✅ Text Similarity
✅ Clustering
✅ Retrieval-Augmented Generation (RAG)

Key Features
Small and Fast: Only 22M parameters, making it efficient for real-time applications.
Good Performance: Balances speed and accuracy, especially for retrieval tasks.
Embedding Size: 384-dimensional vector representation.
Training Data: Trained on 1 billion sentence pairs for semantic similarity.'''


# Step 4 : Store embeddings in FAISS
DB_FAISS_PATH="vectorstore/db_faiss"
db=FAISS.from_documents(text_chunks,embedding_model)
db.save_local(DB_FAISS_PATH)