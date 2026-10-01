from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader("documents/resume.pdf")
docs = loader.load()

print(f"Number of pages loaded: {len(docs)}")
for i, doc in enumerate(docs):
    print(f"\n--- Page {i} ---")
    print(repr(doc.page_content[:200]))