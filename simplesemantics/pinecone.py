from .docstore import DocumentLoader


class Pinecone(DocumentLoader):
    """A version of DocStore compatible with the Pinecone API."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._pinecone = None

    def delete(self, **kwargs):
        raise NotImplementedError

    def fetch(self, ids, **kwargs):
        """The Fetch operation looks up and returns vectors, by ID, from a single namespace. The returned vectors include the vector data and metadata.

        Parameters
        - ids (list of str): A list of IDs to fetch.
        - namespace (str): The namespace to fetch from.
        """
        for document in self.documents:
            if document.document_id in ids:
                yield document

    def query(self, **kwargs):
        """The Query operation returns the IDs of the top-k vectors that are most similar to the query vector.

        Parameters
        - namespace (str): The namespace to query.
        - top_k (int): The number of tokens to return.
        - filter (object): The filter to apply
        - include_values (bool): Whether to include the vector values in the response.
        - include_metadata (bool): Whether to include the vector metadata in the response.
        - vector (list of floats): The query vector.
        - sparse_vector (list of floats): The query sparse vector. Must contain an array of integers named indices and an array of floats named values.
        - id (str): The query ID.
        """
        if not kwargs.get("vector"):
            raise ValueError("Must pass a vector to query")
        query_vector = kwargs.get("vector")
        top_k = kwargs.get("top_k", 10)
        results = []
        results = self.search(query_vector, top_k)
        return results
