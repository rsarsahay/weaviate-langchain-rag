import os
import uuid
from tqdm import tqdm
from weaviate import WeaviateClient
from weaviate.connect import ConnectionParams
from weaviate.classes.config import Configure, Property, DataType
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema.document import Document
import ollama

client = WeaviateClient(
    connection_params=ConnectionParams.from_params(
        http_host="localhost",
        http_port=8080,
        http_secure=False,
        grpc_host="localhost",
        grpc_port=50051,
        grpc_secure=False,
    )
)
print("Connecting to Weaviate...")
print("Weaviate version:", client)
client.connect()

class_name = "class_name"
if not client.collections.exists(class_name):
    client.collections.create(
        name=class_name,
        properties=[
            Property(
                name="content",
                data_type=DataType.TEXT,
            )
        ],
        vectorizer_config=Configure.Vectorizer.none(),
    )

def get_embedding(text: str):
    response = ollama.embeddings(model='nomic-embed-text:latest', prompt=text)
    return response['embedding']

def chunk_text(content: str):
    docs = [Document(page_content=content)]
    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=300,
        chunk_overlap=50
    )
    return splitter.split_documents(docs)

folder_path = "./data"
collection = client.collections.get(class_name)

try:
    for filename in tqdm(os.listdir(folder_path)):
        file_path = os.path.join(folder_path, filename)
        
        if not os.path.isfile(file_path):
            continue
            
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
                chunks = chunk_text(content)
                
                for chunk in chunks:
                    text_chunk = chunk.page_content
                    embedding = get_embedding(text_chunk)
                    collection.data.insert(
                        properties={"content": text_chunk},
                        vector=embedding,
                        uuid=uuid.uuid4()
                    )
        except Exception as e:
            print(f"Error processing file {filename}: {e}")
            continue
            
except Exception as e:
    print(f"Error during processing: {e}")
finally:
    client.close()
