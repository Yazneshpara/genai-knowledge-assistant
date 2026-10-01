import os
from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_pinecone import PineconeVectorStore
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.document_loaders import TextLoader, PyPDFLoader, Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

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

app = FastAPI(title="GenAI Knowledge Assistant")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class Question(BaseModel):
    question: str

def retrieve_context(question, k=5):
    results = vectorstore.similarity_search(question, k=k)
    return [doc.page_content for doc in results]

def generate_answer(question, context_chunks):
    context_text = "\n\n".join(context_chunks)
    messages = prompt.format_messages(context=context_text, question=question)
    response = llm.invoke(messages)
    return response.content

def get_loader_for_file(filepath):
    if filepath.endswith(".pdf"):
        return PyPDFLoader(filepath)
    elif filepath.endswith(".docx"):
        return Docx2txtLoader(filepath)
    elif filepath.endswith(".txt"):
        return TextLoader(filepath)
    else:
        raise ValueError(f"Unsupported file type: {filepath}")

@app.get("/")
def root():
    return {"status": "GenAI Knowledge Assistant is running"}

@app.post("/ask")
def ask(payload: Question):
    chunks = retrieve_context(payload.question)
    answer = generate_answer(payload.question, chunks)
    return {"question": payload.question, "answer": answer, "sources_used": len(chunks)}

@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    save_path = os.path.join("documents", file.filename)
    with open(save_path, "wb") as f:
        content = await file.read()
        f.write(content)

    loader = get_loader_for_file(save_path)
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=100)
    chunks = splitter.split_documents(docs)

    vectorstore.add_documents(chunks)

    return {"filename": file.filename, "chunks_added": len(chunks)}