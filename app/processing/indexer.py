import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from typing import List, Dict
import json
import os
from .video_processor import VideoSegment

class MultiVectorIndexer:
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        self.embedding_model = SentenceTransformer(model_name)
        self.dimension = 384  # Dimension of all-MiniLM-L6-v2 embeddings
        self.index = None
        self.metadata = []
        
    def generate_embeddings(self, segments: List[VideoSegment]) -> np.ndarray:
        """Generate embeddings for text segments"""
        texts = [seg.text for seg in segments]
        return self.embedding_model.encode(texts)
    
    def create_index(self, segments: List[VideoSegment], video_id: str):
        """Create FAISS index with metadata"""
        print("Generating embeddings...")
        embeddings = self.generate_embeddings(segments)
        
        print("Creating FAISS index...")
        self.index = faiss.IndexFlatL2(self.dimension)
        self.index.add(embeddings.astype(np.float32))
        
        # Store metadata
        self.metadata = []
        for i, seg in enumerate(segments):
            self.metadata.append({
                'segment_id': seg.segment_id,
                'start_time': seg.start,
                'end_time': seg.end,
                'text': seg.text,
                'video_id': video_id
            })
        
        print("Saving index...")
        self.save_index(video_id)
    
    def save_index(self, video_id: str):
        """Save index and metadata to disk"""
        os.makedirs('data/indexes', exist_ok=True)
        
        # Save FAISS index
        faiss.write_index(self.index, f'data/indexes/{video_id}.index')
        
        # Save metadata
        with open(f'data/indexes/{video_id}_metadata.json', 'w') as f:
            json.dump(self.metadata, f, indent=2)
    
    def load_index(self, video_id: str) -> bool:
        """Load index and metadata from disk"""
        index_path = f'data/indexes/{video_id}.index'
        metadata_path = f'data/indexes/{video_id}_metadata.json'
        
        if os.path.exists(index_path) and os.path.exists(metadata_path):
            self.index = faiss.read_index(index_path)
            with open(metadata_path, 'r') as f:
                self.metadata = json.load(f)
            return True
        return False
    
    def search(self, query: str, k: int = 5) -> List[Dict]:
        """Search for similar segments"""
        if self.index is None:
            raise ValueError("Index not initialized. Call create_index or load_index first.")
        
        # Generate query embedding
        query_embedding = self.embedding_model.encode([query])
        
        # Search
        distances, indices = self.index.search(query_embedding.astype(np.float32), k)
        
        # Prepare results
        results = []
        for i, idx in enumerate(indices[0]):
            if idx < len(self.metadata):
                result = self.metadata[idx].copy()
                result['score'] = float(distances[0][i])
                results.append(result)
        
        return results