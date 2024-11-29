from langchain_qdrant import Qdrant
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from qdrant_client import QdrantClient, models


from pydantic_settings import BaseSettings, SettingsConfigDict

from fastai.settings import Settings


settings = Settings()

qdrant_api_key = settings.QDRANT_API_KEY
qdrant_url = settings.QDRANT_URL
collection_name = "Websites"

client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key)

# model = ChatGoogleGenerativeAI(
#     model='gemini-pro', google_api_key=settings.GOOGLE_API_KEY,
#     temperatur=0.3, convert_system_message_to_human=True
# )

vector_store = Qdrant(
    client=client,
    collection_name=collection_name,
    embeddings=GoogleGenerativeAIEmbeddings(
        model="models/text-embedding-004",
        task_type="retrieval_document",
        google_api_key=settings.GOOGLE_API_KEY,
    ),
)

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000, chunk_overlap=20, length_function=len
)


def create_collection(collection_name):
    client.create_collection(
        collection_name=collection_name,
        vectors_config=models.VectorParams(size=768, distance=models.Distance.COSINE),
    )
    print(f"Collection {collection_name} created")


def upload_website_to_collection(url: str):
    loader = WebBaseLoader(url)
    docs = loader.load_and_split(text_splitter)
    for doc in docs:
        doc.metadata = {"source_url": url}

    vector_store.add_documents(docs)
    print(f"Successfully uploaded {len(docs)} to vector store")


# embeddings = GoogleGenerativeAIEmbeddings(
#     model="models/text-embedding-004", task_type="retrieval_document"
# )

# create_collection(collection_name)
# upload_website_to_collection('https://www.marxists.org/archive/marx/works/sw/index.htm')
