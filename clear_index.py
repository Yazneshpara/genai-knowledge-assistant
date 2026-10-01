import os
from dotenv import load_dotenv
from pinecone import Pinecone

load_dotenv()

pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index = pc.Index("genai-knowledge-assistant")
index.delete(delete_all=True)
print("Index cleared.")