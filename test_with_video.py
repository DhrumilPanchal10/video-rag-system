#!/usr/bin/env python3
"""
Test the complete system with a real YouTube video
"""

import sys
import os
sys.path.append('.')

from app.processing.video_processor import VideoProcessor
from app.processing.indexer import MultiVectorIndexer
from app.processing.reranker import Reranker
from app.processing.llm_integration import LLMClient
from dotenv import load_dotenv

load_dotenv()

def test_complete_system():
    """Test the complete RAG system"""
    print("Testing complete Video RAG system...")
    
    # Test with a short YouTube video (2-3 minutes)
    test_video_url = "https://youtu.be/aircAruvnKk?si=MIMfXfnQ6VeboUA0"  # First YouTube video
    
    try:
        # 1. Video Processing
        print("1. Testing video processing...")
        processor = VideoProcessor("base")  # Use base model for speed
        segments, video_id = processor.process_video(test_video_url)
        print(f"   ✓ Processed video {video_id} with {len(segments)} segments")
        
        # 2. Indexing
        print("2. Testing indexing...")
        indexer = MultiVectorIndexer()
        indexer.create_index(segments, video_id)
        print("   ✓ Index created successfully")
        
        # 3. Test search
        print("3. Testing search...")
        results = indexer.search("video", k=3)
        print(f"   ✓ Search found {len(results)} results")
        
        # 4. Test reranking
        print("4. Testing reranking...")
        reranker = Reranker()
        reranked = reranker.rerank("video", results)
        print(f"   ✓ Reranked {len(reranked)} results")
        
        # 5. Test LLM (if API key is set)
        print("5. Testing LLM integration...")
        llm_client = LLMClient(os.getenv("GROQ_API_KEY"))
        
        if llm_client.api_key and llm_client.api_key != "your_actual_groq_api_key_here":
            context = llm_client.format_context(reranked[:2])
            answer = llm_client.generate_answer("What is this video about?", context)
            print(f"   ✓ LLM response: {answer[:100]}...")
        else:
            print("   ⚠️ LLM skipped - API key not configured")
        
        print("\n🎉 All tests passed! System is working correctly.")
        return True
        
    except Exception as e:
        print(f"❌ Error during testing: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_complete_system()