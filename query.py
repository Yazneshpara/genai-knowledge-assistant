import os
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_pinecone import PineconeVectorStore
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

INDEX_NAME = "genai-knowledge-assistant"

embeddings = OpenAIEmbeddings(model="text-embedding-3-small", api_key=os.getenv("OPENAI_API_KEY"))
vectorstore = PineconeVectorStore(index_name=INDEX_NAME, embedding=embeddings)
llm = ChatOpenAI(model="gpt-4o-mini", api_key=os.getenv("OPENAI_API_KEY"))

prompt = ChatPromptTemplate.from_template("""Answer the question using ONLY the context below. Be thorough — if the context lists multiple points relevant to the question, include ALL of them, not just one. If the answer isn't in the context, say "I don't know based on the provided documents."

Context:
{context}

Question: {question}

Answer:""")

def retrieve_context(question, k=5):
    results = vectorstore.similarity_search(question, k=k)
    return [doc.page_content for doc in results]

def generate_answer(question, context_chunks):
    context_text = "\n\n".join(context_chunks)
    messages = prompt.format_messages(context=context_text, question=question)
    response = llm.invoke(messages)
    return response.content

if __name__ == "__main__":
    while True:
        question = input("\nAsk a question about SmartPay (or type 'exit'): ")
        if question.lower() == "exit":
            break
        chunks = retrieve_context(question)
        print("\n--- Retrieved chunks ---")
        for i, c in enumerate(chunks):
            print(f"[{i}] {c}\n")
        answer = generate_answer(question, chunks)
        print("\nAnswer:", answer)