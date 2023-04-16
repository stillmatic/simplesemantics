import sssdb


def test_init():
    assert sssdb.__version__ == "0.1.0"


class TestModel:
    def encode(self, text):
        return [1, 2, 3]


def test_basics():
    document_name = "Document1"
    document_content = "My documents are in the box"
    dl = sssdb.DocumentStore(dense_model=TestModel())
    dl.load_document(document_name=document_name, document_content=document_content)
    document = dl.documents[0]
    assert document.document_name == document_name
    assert document.document_content == document_content
    assert document.document_id == 0
    assert document.dense_embedding is not None
    assert document.metadata == {}

    # test save
    dl.save("test.json")

    # test load
    dl2 = sssdb.DocumentStore(dense_model=TestModel())
    dl2.load("test.json")
    document = dl2.documents[0]
    assert document.document_name == document_name
    assert document.document_content == document_content
    assert document.document_id == 0
    assert document.dense_embedding is not None
    assert document.metadata == {}


def test_dummy_example():
    dense_model = TestModel()
    sparse_model = TestModel()
    dl = sssdb.DocumentStore(dense_model=dense_model, sparse_model=sparse_model)
    dl.load_document("Document1", "My documents are in the box")
    dl.load_document("Document2", "I like to eat apples and oranges")
    dl.load_document(
        "Document3",
        "Processing and storing tabular datasets, e.g. from CSV or Parquet files",
    )
    dl.load_document("Document4", "Decorate with pictures you love.")
    dl.load_document(
        "Document5", "Lorem ipsum dolor sit amet, consectetur adipiscing elit."
    )
    dl.load_document(
        "Document6",
        "Fiberboard is a stable and durable material made of leftover wood from the wood industry.",
    )
    results = dl.search("Where are my documents?", 5)
    print(results)
    assert len(results) == 5
    # results should just be ordered by document_id since all embeddings are the same
    assert results[0][0] == 1.0
    assert results[0][1].document_name == "Document1"


def test_example_with_minilm():
    from sentence_transformers import SentenceTransformer
    from sssdb import wrappers, DocumentStore

    dense_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    sparse_model = wrappers.SpladeWrapper(
        "naver/splade-cocondenser-ensembledistil", agg="mean"
    )

    doc_loader = DocumentStore(dense_model=dense_model, sparse_model=sparse_model)
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
    # stochastic, so we can't assert the exact results
    assert len(results) == 5
    assert results[0][1].document_name == "Document1"
    assert results[0][0] > 0.8850
    assert results[1][1].document_name == "Document3"
    assert results[1][0] < 0.4
