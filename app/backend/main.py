from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import os
from datetime import datetime

# Add parent directory to path for imports
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from processing.video_processor import VideoProcessor
from processing.indexer import MultiVectorIndexer
from processing.reranker import Reranker
from processing.llm_integration import LLMClient

app = FastAPI(title="Video RAG API", version="1.0.0")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize components
video_processor = VideoProcessor()
indexer = MultiVectorIndexer()
reranker = Reranker()
llm_client = LLMClient()

class VideoProcessRequest(BaseModel):
    video_url: str

class QueryRequest(BaseModel):
    video_id: str
    query: str
    top_k: Optional[int] = 5
    rerank_top_k: Optional[int] = 3

class ProcessedVideoResponse(BaseModel):
    video_id: str
    segments: List[dict]
    message: str

class QueryResponse(BaseModel):
    answer: str
    relevant_segments: List[dict]
    timestamps: List[float]

@app.post("/process_video", response_model=ProcessedVideoResponse)
async def process_video(request: VideoProcessRequest):
    """Process a video URL and create search index"""
    try:
        segments, video_id = video_processor.process_video(request.video_url)
        indexer.create_index(segments, video_id)
        
        return ProcessedVideoResponse(
            video_id=video_id,
            segments=[seg.dict() for seg in segments],
            message="Video processed successfully"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing video: {str(e)}")

@app.post("/query", response_model=QueryResponse)
async def query_video(request: QueryRequest):
    """Query a processed video"""
    try:
        # Load index if not already loaded
        if not indexer.load_index(request.video_id):
            raise HTTPException(status_code=404, detail="Video not processed. Please process it first.")
        
        # Initial vector search
        initial_results = indexer.search(request.query, k=request.top_k)
        
        # Rerank results
        reranked_results = reranker.rerank(request.query, initial_results)[:request.rerank_top_k]
        
        # Generate answer with LLM
        context = llm_client.format_context(reranked_results)
        answer = llm_client.generate_answer(request.query, context)
        
        # Extract timestamps from answer
        timestamps = llm_client.extract_timestamps(answer)
        
        return QueryResponse(
            answer=answer,
            relevant_segments=reranked_results,
            timestamps=timestamps
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error querying video: {str(e)}")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "timestamp": datetime.now()}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)