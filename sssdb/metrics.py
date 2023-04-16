import numpy as np


def cosine_similarity(vec1, vec2) -> float:
    """Computes the cosine similarity between two vectors"""
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))


def euclidean_distance(vec1, vec2) -> float:
    """Computes the euclidean distance between two vectors"""
    return np.linalg.norm(vec1 - vec2)
