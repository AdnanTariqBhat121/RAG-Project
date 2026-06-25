from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_classic.retrievers.multi_query import MultiQueryRetriever
from langchain_mistralai import ChatMistralAI

from dotenv import load_dotenv
load_dotenv()

docs = [
    Document(page_content="Gradient Descent is an optimization algorithm used in Machine Learning"),
    Document(page_content="Gradient Descent minimizes the loss function"),
    Document(page_content="Gradient Descent is an optimization that minimizes the loss funcion"),
    Document(page_content="Neural Network use Gradient Descent for training"),
    Document(page_content="Support Vector Machines are supervised learning algorithms")
]

embeddings = HuggingFaceBgeEmbeddings(model_name = "sentence-transformers/all-MiniLM-L6-v2")
vectorstore = Chroma.from_documents(docs, embeddings)

retriever = vectorstore.as_retriever()

llm = ChatMistralAI(model="mistral-small-latest")

multi_query_retriever = MultiQueryRetriever.from_llm(
    retriever = retriever,
    llm = llm
)

query = "What is Gradient Descent?"

docs = multi_query_retriever.invoke(query)

print("\nRetrieved Documents:\n")
for doc in docs:
    print(doc.page_content)