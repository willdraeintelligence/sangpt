from langchain_community.vectorstores import FAISS
from langchain_community.llms import Ollama
from sentence_transformers import SentenceTransformer

embeddings = SentenceTransformer("all-MiniLM-L6-v2")

vectorstore = FAISS.load_local("vectorstore", embeddings)

llm = Ollama(model="mistral")

def ask_question(query):
    docs = vectorstore.similarity_search(query, k=3)

    context = "\n".join([doc.page_content for doc in docs])

    prompt = f"""
    Use the context to answer the question.

    Context:
    {context}

    Question:
    {query}
    """

    response = llm.invoke(prompt)
    return response