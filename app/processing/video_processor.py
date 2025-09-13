import os
import yt_dlp
import whisper
from typing import List, Dict, Tuple
import json
from datetime import datetime
import spacy
from pydantic import BaseModel

class VideoSegment(BaseModel):
    text: str
    start: float
    end: float
    segment_id: int

class VideoProcessor:
    def __init__(self, model_size="base"):
        self.model = whisper.load_model(model_size)
        self.nlp = spacy.load("en_core_web_sm")
        
    def download_video_audio(self, video_url: str, output_path: str = "data/videos") -> Tuple[str, str]:
        """Download audio from YouTube video"""
        os.makedirs(output_path, exist_ok=True)
        
        ydl_opts = {
            'format': 'bestaudio/best',
            'outtmpl': f'{output_path}/%(id)s.%(ext)s',
            'postprocessors': [{
                'key': 'FFmpegExtractAudio',
                'preferredcodec': 'mp3',
                'preferredquality': '192',
            }],
            'quiet': True
        }
        
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(video_url, download=True)
            video_id = info['id']
            audio_path = f"{output_path}/{video_id}.mp3"
            
        return audio_path, video_id
    
    def transcribe_audio(self, audio_path: str) -> Dict:
        """Transcribe audio using Whisper"""
        result = self.model.transcribe(audio_path, word_timestamps=True)
        return result
    
    def semantic_chunking(self, transcription: Dict, max_words: int = 150) -> List[VideoSegment]:
        """Chunk transcription semantically using spaCy"""
        segments = []
        
        # Create segments based on Whisper's segments
        for i, segment in enumerate(transcription['segments']):
            segments.append(VideoSegment(
                text=segment['text'].strip(),
                start=segment['start'],
                end=segment['end'],
                segment_id=i
            ))
        
        # Further chunk based on semantic boundaries
        semantic_segments = []
        current_segment = []
        current_start = 0
        current_end = 0
        segment_id = 0
        
        for segment in segments:
            doc = self.nlp(segment.text)
            sentences = [sent.text.strip() for sent in doc.sents]
            
            for sentence in sentences:
                words = sentence.split()
                
                # If adding this sentence would exceed max words, save current segment
                if len(current_segment) + len(words) > max_words and current_segment:
                    semantic_segments.append(VideoSegment(
                        text=" ".join(current_segment),
                        start=current_start,
                        end=current_end,
                        segment_id=segment_id
                    ))
                    segment_id += 1
                    current_segment = []
                
                # Add sentence to current segment
                if not current_segment:
                    current_start = segment.start
                
                current_segment.append(sentence)
                current_end = segment.end
        
        # Add the last segment if it exists
        if current_segment:
            semantic_segments.append(VideoSegment(
                text=" ".join(current_segment),
                start=current_start,
                end=current_end,
                segment_id=segment_id
            ))
        
        return semantic_segments
    
    def process_video(self, video_url: str) -> Tuple[List[VideoSegment], str]:
        """Full processing pipeline for a video"""
        print("Downloading video audio...")
        audio_path, video_id = self.download_video_audio(video_url)
        
        print("Transcribing audio...")
        transcription = self.transcribe_audio(audio_path)
        
        print("Chunking transcription semantically...")
        segments = self.semantic_chunking(transcription)
        
        # Save segments to JSON
        os.makedirs("data/transcripts", exist_ok=True)
        output_path = f"data/transcripts/{video_id}.json"
        with open(output_path, 'w') as f:
            json.dump([seg.dict() for seg in segments], f, indent=2)
        
        return segments, video_id