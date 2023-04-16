import heapq
from typing import List, Optional, Tuple, Any
import json
import numpy as np
from sssdb.document import Document
from sssdb.metrics import cosine_similarity, euclidean_distance


class DocumentStore:
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
        dense_embedding: Optional[np.ndarray] = None,
        sparse_embedding: Optional[np.ndarray] = None,
        split_lines: bool = False,
        metadata: Optional[dict] = None,
    ) -> None:
        """Loads a document into the DocumentLoader.

        - document_name: str, name of the document
        - document_content: str, content of the document
        - split_lines: bool, whether to split the document content by line
        - metadata: dict, metadata to store with the document
        """
        if split_lines:
            lines = document_content.split("\n")
        else:
            lines = [document_content]
        max_document_id = len(self.documents)
        for index, line in enumerate(lines):
            # Use a dictionary to store embeddings conditionally
            embeddings = {}
            if self.dense_model and not dense_embedding:
                embeddings["dense"] = self.dense_model.encode(line)
            if self.sparse_model and not sparse_embedding:
                embeddings["sparse"] = self.sparse_model.encode(line)
            if dense_embedding:
                embeddings["dense"] = dense_embedding
            if sparse_embedding:
                embeddings["sparse"] = sparse_embedding
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
        k: int = 10,
        alpha: Optional[float] = None,
        metadata: Optional[dict] = None,
        metric: str = "cosine",
    ) -> List[Tuple[float, Document]]:
        """Searches the DocumentLoader for the top k documents that match the query

        - query: str, query to search for
        - k: int, number of documents to return
        - alpha: float, weight of dense model
        - metadata: dict, metadata to filter documents by
        Returns a list of tuples of the form (score, document)
        """
        if alpha is None:
            alpha = self.alpha
        embeddings = {}
        if self.dense_model:
            embeddings["dense"] = self.dense_model.encode(query)
        if self.sparse_model:
            embeddings["sparse"] = self.sparse_model.encode(query)
        if metric == "cosine":
            distance_fn = cosine_similarity
        elif metric == "euclidean":
            distance_fn = euclidean_distance

        def score(document: Document) -> Optional[float]:
            dense_score = sparse_score = 0
            if metadata:
                for key, value in metadata.items():
                    if document.metadata.get(key) != value:
                        return None
            if self.dense_model:
                dense_score = distance_fn(embeddings["dense"], document.dense_embedding)
            if self.sparse_model:
                sparse_score = distance_fn(
                    embeddings["sparse"], document.sparse_embedding
                )
            return alpha * dense_score + (1 - alpha) * sparse_score

        return heapq.nlargest(
            k,
            ((score(document), document) for document in self.documents),
            key=lambda x: x[0],
        )

    def save(self, outfile: str):
        with open(outfile, "w") as f:
            for document in self.documents:
                f.write(json.dumps(document.serialize()) + "\n")

    def load(self, infile: str):
        with open(infile, "r") as f:
            doc = f.readline()
            while doc:
                self.documents.append(Document(**json.loads(doc)))
                doc = f.readline()
