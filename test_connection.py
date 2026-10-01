from dotenv import load_dotenv
import os
from openai import OpenAI
from pinecone import Pinecone

load_dotenv()

# --- Test OpenAI ---
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
response = openai_client.embeddings.create(
    input="Hello world",
    model="text-embedding-3-small"
)
print("OpenAI connected! Embedding length:", len(response.data[0].embedding))

# --- Test Pinecone ---
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
print("Pinecone connected! Existing indexes:", pc.list_indexes())