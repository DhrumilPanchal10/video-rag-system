import streamlit as st
import requests
import json
from datetime import timedelta
import time

# Configuration
BACKEND_URL = "http://localhost:8000"

st.set_page_config(
    page_title="Video RAG System",
    page_icon="🎥",
    layout="wide"
)

st.title("🎥 Real-Time Adaptive RAG System for Long-Form Video QA")

# Initialize session state
if "processed_videos" not in st.session_state:
    st.session_state.processed_videos = {}
if "current_video_id" not in st.session_state:
    st.session_state.current_video_id = None

# Sidebar for video processing
with st.sidebar:
    st.header("Video Processing")
    
    video_url = st.text_input("Enter YouTube Video URL:", "")
    
    if st.button("Process Video"):
        if video_url:
            with st.spinner("Processing video (this may take a while)..."):
                try:
                    response = requests.post(
                        f"{BACKEND_URL}/process_video",
                        json={"video_url": video_url}
                    )
                    
                    if response.status_code == 200:
                        result = response.json()
                        video_id = result["video_id"]
                        st.session_state.processed_videos[video_id] = result
                        st.session_state.current_video_id = video_id
                        st.success(f"Video processed successfully! ID: {video_id}")
                    else:
                        st.error(f"Error processing video: {response.json()['detail']}")
                except Exception as e:
                    st.error(f"Error connecting to backend: {str(e)}")
        else:
            st.warning("Please enter a YouTube URL")

    # Display processed videos
    if st.session_state.processed_videos:
        st.header("Processed Videos")
        for video_id in st.session_state.processed_videos:
            if st.button(f"Select: {video_id}", key=f"btn_{video_id}"):
                st.session_state.current_video_id = video_id

# Main content area
if st.session_state.current_video_id:
    video_id = st.session_state.current_video_id
    video_data = st.session_state.processed_videos[video_id]
    
    st.header(f"Video ID: {video_id}")
    st.write(f"Number of segments: {len(video_data['segments'])}")
    
    # Query section
    st.subheader("Ask a Question")
    query = st.text_input("Enter your question about the video:")
    
    if st.button("Get Answer") and query:
        with st.spinner("Searching for relevant content and generating answer..."):
            start_time = time.time()
            
            try:
                response = requests.post(
                    f"{BACKEND_URL}/query",
                    json={
                        "video_id": video_id,
                        "query": query,
                        "top_k": 10,
                        "rerank_top_k": 5
                    }
                )
                
                if response.status_code == 200:
                    result = response.json()
                    end_time = time.time()
                    
                    # Display answer
                    st.subheader("Answer:")
                    st.write(result["answer"])
                    
                    # Display latency
                    st.write(f"Response time: {end_time - start_time:.2f} seconds")
                    
                    # Display relevant segments
                    st.subheader("Relevant Video Segments:")
                    for i, segment in enumerate(result["relevant_segments"]):
                        with st.expander(f"Segment {i+1} (Time: {segment['start_time']:.2f}s - {segment['end_time']:.2f}s, Score: {segment['rerank_score']:.4f})"):
                            st.write(segment["text"])
                    
                    # Display extracted timestamps
                    if result["timestamps"]:
                        st.subheader("Key Timestamps:")
                        for ts in result["timestamps"]:
                            st.write(f"- {timedelta(seconds=int(ts))}")
                
                else:
                    st.error(f"Error querying video: {response.json()['detail']}")
            
            except Exception as e:
                st.error(f"Error connecting to backend: {str(e)}")

else:
    st.info("Please process a YouTube video using the sidebar to get started.")
    st.video("https://www.youtube.com/watch?v=dQw4w9WgXcQ")  # Example video

# Footer
st.markdown("---")
st.markdown("### Real-Time Adaptive RAG System for Long-Form Video QA")
st.markdown("Built with Whisper, FAISS, Sentence Transformers, and Groq API")