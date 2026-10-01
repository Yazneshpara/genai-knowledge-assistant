# GenAI Knowledge Assistant

A Retrieval-Augmented Generation (RAG) platform that answers questions grounded in your own uploaded documents — built with LangChain, OpenAI, Pinecone, and FastAPI, containerized with Docker and deployable on Kubernetes.

## Features

- **Document ingestion**: upload `.txt`, `.pdf`, or `.docx` files through a web UI or API
- **RAG pipeline**: chunks documents, embeds them with OpenAI, stores them in Pinecone, and retrieves the most relevant chunks for any question
- **Grounded answers**: uses GPT-4o-mini to answer strictly from retrieved context, reducing hallucination
- **Chat UI**: a simple ChatGPT-style web interface (`chat.html`)
- **Containerized**: ships with a Dockerfile for consistent deployment anywhere
- **Kubernetes-ready**: includes a Deployment + Service manifest for running on any Kubernetes cluster

## Tech Stack

LangChain · OpenAI API · Pinecone · FastAPI · Docker · Kubernetes

## Architecture

1. **Ingest** (`ingest.py`) — loads documents, splits them into chunks, embeds them, and stores them in Pinecone
2. **Query** (`query.py`) — CLI tool to ask questions against the knowledge base from the terminal
3. **API** (`main.py`) — FastAPI server exposing `/ask` and `/upload` endpoints
4. **UI** (`chat.html`) — browser-based chat interface that calls the API

## Setup

1. Clone the repo and create a virtual environment:

   python3 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt

2. Create a `.env` file with your API keys:

   OPENAI_API_KEY=your_key_here
   PINECONE_API_KEY=your_key_here

3. Ingest the sample documents:

   python ingest.py

4. Run the API:

   uvicorn main:app --reload

5. Open `chat.html` in your browser to use the chat interface.

## Running with Docker

docker build -t genai-knowledge-assistant .
docker run -p 8000:8000 --env-file .env genai-knowledge-assistant

## Running on Kubernetes (Minikube)

minikube start
minikube image load genai-knowledge-assistant:latest
kubectl create secret generic genai-secrets --from-env-file=.env
kubectl apply -f deployment.yaml
minikube service genai-knowledge-assistant-service --url

## Example

**Question:** "What are the drawbacks of SmartPay?"

**Answer:** Retrieves relevant chunks from the knowledge base and generates a grounded response citing privacy concerns, accessibility issues, and centralization risk — without hallucinating information not present in the source documents.