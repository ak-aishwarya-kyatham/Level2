import logging
import numpy as np
from typing import List, Tuple

logger = logging.getLogger(__name__)

class DuplicateDetectionAgent:
    """
    Detects duplicate news using Cosine Similarity.
    """
    def __init__(self, similarity_threshold: float = 0.85):
        self.similarity_threshold = similarity_threshold

    def cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        v1 = np.array(vec1)
        v2 = np.array(vec2)
        if np.linalg.norm(v1) == 0 or np.linalg.norm(v2) == 0:
            return 0.0
        return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

    async def run(self, new_embedding: List[float], existing_embeddings: List[List[float]]) -> Tuple[bool, float]:
        """
        Returns a tuple of (is_duplicate, max_similarity_score).
        """
        if not new_embedding or not existing_embeddings:
            return False, 0.0
            
        logger.info(f"Checking duplicate against {len(existing_embeddings)} existing documents...")
        
        max_similarity = 0.0
        for emb in existing_embeddings:
            sim = self.cosine_similarity(new_embedding, emb)
            if sim > max_similarity:
                max_similarity = sim
                
        is_duplicate = max_similarity >= self.similarity_threshold
        if is_duplicate:
            logger.info(f"Duplicate detected with score {max_similarity:.4f}")
        else:
            logger.info(f"No duplicate found. Max score: {max_similarity:.4f}")
            
        return is_duplicate, max_similarity
