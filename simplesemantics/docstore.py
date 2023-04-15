import heapq
from typing import List, Optional, Tuple, Any
from sentence_transformers import SentenceTransformer
import numpy as np


class Document:
    def __init__(
        self,
        document_id: int,
        document_name: str,
        document_content: str,
        dense_embedding: np.ndarray,
        sparse_embedding: np.ndarray,
        metadata: dict,
    ):
        self.document_name = document_name
        self.document_content = document_content
        self.document_id = document_id
        self.dense_embedding = dense_embedding
        self.sparse_embedding = sparse_embedding
        self.metadata = metadata

    def __repr__(self):
        return f"Document({self.document_id}, {self.document_name}, {self.document_content})"

    def __lt__(self, other):
        return self.document_id < other.document_id

    def __le__(self, other):
        return self.document_id <= other.document_id


class DocumentLoader:
    def __init__(
        self, alpha: float = 0.7, dense_model: Any = None, sparse_model: Any = None
    ):
        """Instantiates a new DocumentLoader

        - alpha: float, weight of dense model
        - dense_model: Any, dense model to use for encoding
        - sparse_model: Any, sparse model to use for encoding
        Must pass at least one model. Alpha determines the weight of the dense model.
        """
        self.documents = []
        # must have at least one model
        assert dense_model is not None or sparse_model is not None
        # model must have a encode method to return embeddings
        if dense_model is not None:
            assert hasattr(dense_model, "encode")
        self.dense_model = dense_model
        if sparse_model is not None:
            assert hasattr(sparse_model, "encode")
        self.sparse_model = sparse_model
        self.alpha = alpha

    def load_document(
        self,
        document_name: str,
        document_content: str,
        metadata: Optional[dict] = None,
    ) -> None:
        lines = document_content.split("\n")
        max_document_id = len(self.documents)
        for index, line in enumerate(lines):
            # Use a dictionary to store embeddings conditionally
            embeddings = {}
            if self.dense_model:
                embeddings["dense"] = self.dense_model.encode(line)
            if self.sparse_model:
                embeddings["sparse"] = self.sparse_model.encode(line)
            if metadata is None:
                metadata = {}
            doc = Document(
                document_id=max_document_id + index,
                document_name=document_name,
                document_content=line,
                dense_embedding=embeddings.get("dense"),
                sparse_embedding=embeddings.get("sparse"),
                metadata=metadata,
            )
            self.documents.append(doc)

    def search(
        self,
        query: str,
        k: int,
        alpha: Optional[float] = None,
        metadata: Optional[dict] = None,
    ) -> List[Tuple[str, str]]:
        if alpha is None:
            alpha = self.alpha
        embeddings = {}
        if self.dense_model:
            embeddings["dense"] = self.dense_model.encode(query)
        if self.sparse_model:
            embeddings["sparse"] = self.sparse_model.encode(query)

        def score(document: Document) -> Optional[float]:
            dense_score = sparse_score = 0
            if metadata:
                for key, value in metadata.items():
                    if document.metadata.get(key) != value:
                        return None
            if self.dense_model:
                dense_score = self._cosine_similarity(
                    embeddings["dense"], document.dense_embedding
                )
            if self.sparse_model:
                sparse_score = self._cosine_similarity(
                    embeddings["sparse"], document.sparse_embedding
                )
            return alpha * dense_score + (1 - alpha) * sparse_score

        return heapq.nlargest(
            k,
            ((score(document), document) for document in self.documents),
            key=lambda x: x[0],
        )

    @staticmethod
    def _cosine_similarity(vec1, vec2) -> float:
        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        norm1 = sum(a * a for a in vec1) ** 0.5
        norm2 = sum(b * b for b in vec2) ** 0.5
        return dot_product / (norm1 * norm2)


if __name__ == "__main__":
    dense_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    sparse_model = SpladeWrapper("naver/splade-cocondenser-ensembledistil", agg="mean")

    doc_loader = DocumentLoader(dense_model=dense_model, sparse_model=sparse_model)
    doc_loader.load_document("Document1", "This is a relevant document to the query")
    doc_loader.load_document("Document2", "I like to eat apples and oranges")
    doc_loader.load_document("Document3", "I like to eat apples and oranges")
    doc_loader.load_document("Document4", "I like to eat apples and oranges")
    doc_loader.load_document("Document5", "I like to eat apples and oranges")
    doc_loader.load_document("Document6", "I like to eat apples and oranges")
    results = doc_loader.search("Where are my documents?", 5)
