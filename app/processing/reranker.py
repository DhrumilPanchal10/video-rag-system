from sentence_transformers import CrossEncoder
from typing import List, Dict

class Reranker:
    def __init__(self, model_name='cross-encoder/ms-marco-MiniLM-L-6-v2'):
        self.model = CrossEncoder(model_name, max_length=512)
    
    def rerank(self, query: str, passages: List[Dict]) -> List[Dict]:
        """Rerank passages based on relevance to query"""
        if not passages:
            return []
        
        # Prepare pairs for cross-encoder
        pairs = [(query, passage['text']) for passage in passages]
        
        # Get scores
        scores = self.model.predict(pairs)
        
        # Add scores to passages and sort
        for i, passage in enumerate(passages):
            passage['rerank_score'] = float(scores[i])
        
        # Sort by rerank score (descending)
        reranked = sorted(passages, key=lambda x: x['rerank_score'], reverse=True)
        
        return reranked