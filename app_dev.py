from dotenv import load_dotenv
load_dotenv()
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# 1. Document Loading
data = PyPDFLoader("document loaders/attention-is-all-you-need-Paper.pdf")
docs = data.load()

# 2. Splitter instantiation
splitter = RecursiveCharacterTextSplitter(
    chunk_size = 1500,
    chunk_overlap = 150
)

# Document Splitting
chunks = splitter.split_documents(docs)

# 3. Embedding Model
embedding = HuggingFaceEmbeddings(
    model_name = "all-MiniLM-L6-v2"
)

# 4. Embedding & db CREATION
vector_store = Chroma.from_documents(
    documents = chunks,
    embedding = embedding,
    persist_directory = "ChromaDb"
    
)

# 5. Retriever Creation
retriever = vector_store.as_retriever(
    search_type =  "mmr",
    search_kwargs = { 
    "k" : 5,
    "fetch_k" : 20}
)

result = retriever.invoke("What is the Transformer")
print(result)