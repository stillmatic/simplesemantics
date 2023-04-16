from sentence_transformers import SentenceTransformer
from simplesemantics import wrappers, DocumentLoader

if __name__ == "__main__":
    dense_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    sparse_model = wrappers.SpladeWrapper(
        "naver/splade-cocondenser-ensembledistil", agg="mean"
    )

    doc_loader = DocumentLoader(dense_model=dense_model, sparse_model=sparse_model)
    doc_loader.load_document("Document1", "My documents are in the box")
    doc_loader.load_document("Document2", "I like to eat apples and oranges")
    doc_loader.load_document(
        "Document3",
        "Processing and storing tabular datasets, e.g. from CSV or Parquet files",
    )
    doc_loader.load_document("Document4", "Decorate with pictures you love.")
    doc_loader.load_document(
        "Document5", "Lorem ipsum dolor sit amet, consectetur adipiscing elit."
    )
    doc_loader.load_document(
        "Document6",
        "Fiberboard is a stable and durable material made of leftover wood from the wood industry.",
    )
    results = doc_loader.search("Where are my documents?", 5)
    print(results)
