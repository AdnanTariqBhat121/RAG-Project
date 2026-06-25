from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceBgeEmbeddings

docs = [
    Document(page_content="Gradient Descent is an optimization algorithm used in Machine Learning"),
    Document(page_content="Gradient Descent minimizes the loss function"),
    Document(page_content="Gradient Descent is an optimization that minimizes the loss funcion"),
    Document(page_content="Neural Network use Gradient Descent for training"),
    Document(page_content="Support Vector Machines are supervised learning algorithms")
]

embedding_model = HuggingFaceBgeEmbeddings(model_name = "sentence-transformers/all-MiniLM-L6-v2")

vectorestore = Chroma.from_documents(docs,embedding_model)

similarity_retriever = vectorestore.as_retriever(
    search_type = "similarity",
    search_kwargs = {"k":3}
)

print("\n======== Similarity Search Results========= \n")

similarity_docs = similarity_retriever.invoke("What is Gradient Descent?")

for doc in similarity_docs:
    print(doc.page_content)

mmr_retriever = vectorestore.as_retriever(
    search_type = "mmr",
    search_kwargs = {"k":3}
)

print("\n ========= MMR Results ========= \n")
mmr_docs = mmr_retriever.invoke("What is Gradient Descent")

for doc in mmr_docs:
    print(doc.page_content)