from langchain_community.document_loaders import PyMuPDFLoader

def load_pdf(file_path):
    loader = PyMuPDFLoader(file_path)
    documents = loader.load()
    print(f"[INFO] loaded {len(documents)} documents from {file_path} successfully")
    return documents


if __name__ == "__main__":
    docs = load_pdf("../data/sample.pdf")

    print(f"Total documents/pages loaded: {len(docs)}")

    print("\nFirst document:\n")

    print(docs[0])