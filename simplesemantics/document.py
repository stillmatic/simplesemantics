from typing import List, Optional, Tuple, Any
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

    def serialize(self):
        return {
            "document_name": self.document_name,
            "document_content": self.document_content,
            "document_id": self.document_id,
            "dense_embedding": self.dense_embedding,
            "sparse_embedding": self.sparse_embedding,
            "metadata": self.metadata,
        }
