from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from sentence_transformers import SentenceTransformer

loader = PyPDFLoader("data/document.pdf")
docs = loader.load()

splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = splitter.split_documents(docs)

embeddings = SentenceTransformer("all-MiniLM-L6-v2")

vectorstore = FAISS.from_documents(chunks, embeddings)
vectorstore.save_local("vectorstore")