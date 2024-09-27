from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama
# from llama_index import Document, VectorStoreIndex
import requests


url = "http://iotekno.id:8891/media/upload/documents/Materi_Django.pdf"  # Replace with your URL
response = requests.get(url)
document_text = response.text
# document = Document(document_text)

documents = SimpleDirectoryReader("media").load_data()
Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-base-en-v1.5")
Settings.llm = Ollama(model="llama3.1", request_timeout=360.0)
index = VectorStoreIndex.from_documents(
    documents,
)

# index = VectorStoreIndex.from_documents([document])

query_engine = index.as_query_engine()
response = query_engine.query("apa itu gundam?")
print(response)
