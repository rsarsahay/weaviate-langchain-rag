from langchain_ollama import OllamaLLM
from langchain.prompts import PromptTemplate
from weaviate import WeaviateClient
from weaviate.connect import ConnectionParams
import ollama
import time


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
client.connect()


def get_embedding(text):
    try:
        response = ollama.embeddings(model='nomic-embed-text:latest', prompt=text)
        return response['embedding']
    except Exception as e:
        print("Embedding error:", e)
        return None


def query_weaviate_hybrid(query_text, alpha=0.7, limit=3):
    vector = get_embedding(query_text)
    print("Vector:", vector)    
    if not vector:
        return []

    try:
        collection = client.collections.get("class_name")
        results = collection.query.hybrid(
            query=query_text,
            vector=vector,
            alpha=alpha,
            limit=limit,
            # offset=offset,
            return_metadata=["score"]
        )

        docs = []
        if not results.objects:
            print("No results found.")
            return []

        for i, obj in enumerate(results.objects, 1):
            content = obj.properties.get("content", "")
            score = getattr(obj.metadata, "score", 0)
            print(f"\n--- Document {i} (score: {score:.4f}) ---\n{content[:300]}...")
            docs.append(content)
        return docs
    except Exception as e:
        print("Error during hybrid search:", e)
        return []


llm = OllamaLLM(model="granite3.3:2b" )

prompt_template = PromptTemplate(
    input_variables=["context", "question"],
    template="""
    You are an expert assistant helping users extract answers from documents.

    Use only the provided context to answer the question. 
    If the answer is not found in the context, respond with: "Answer is not available in the context."
    Answer in detail.

Context:
{context}

Question: {question}

Answer:
""".strip()
)

# Using modern LCEL (LangChain Expression Language) instead of deprecated LLMChain(prompt, llm)
qa_chain = prompt_template | llm

def rag_pipeline(question):
    documents = query_weaviate_hybrid(question)
    if not documents:
        return "No relevant documents found."

    combined_context = "\n\n".join(documents)
    try:
        # Updated to use LCEL invoke method
        response = qa_chain.invoke({
            "context": combined_context, 
            "question": question
        })
        return response
    except Exception as e:
        return f"LLM generation error: {e}"


if __name__ == "__main__":
    while True:
        try:
            start_time = time.time()
            question = input("Enter your question: ").strip()
            answer = rag_pipeline(question)
            print("\n Answer :\n", answer)
            end_time = time.time()
            print(f"\nTime taken: {round(end_time - start_time, 2)} seconds")
        finally:
            client.close()
