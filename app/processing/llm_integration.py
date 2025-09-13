import requests
import json
from typing import List, Dict
import re
import os

class LLMClient:
    def __init__(self, api_key: str = None, model: str = "llama3-70b-8192"):
        self.api_key = api_key or os.getenv("GROQ_API_KEY")
        self.model = model
        self.base_url = "https://api.groq.com/openai/v1/chat/completions"
    
    def format_context(self, passages: List[Dict]) -> str:
        """Format retrieved passages into context for LLM"""
        context = ""
        for i, passage in enumerate(passages):
            context += f"[Segment {i+1}, Time: {passage['start_time']:.2f}-{passage['end_time']:.2f}s]: {passage['text']}\n\n"
        return context
    
    def generate_answer(self, query: str, context: str) -> str:
        """Generate answer using LLM with provided context"""
        prompt = f"""Based on the following video content, answer the question comprehensively. 
        Always cite the specific timestamps where the information comes from.

        Video Content:
        {context}

        Question: {query}

        Answer:"""
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "system",
                    "content": "You are a helpful assistant that answers questions based on video content. Always cite timestamps from the video to support your answers."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "temperature": 0.1,
            "max_tokens": 1024
        }
        
        try:
            response = requests.post(self.base_url, headers=headers, json=payload)
            response.raise_for_status()
            result = response.json()
            return result['choices'][0]['message']['content']
        except Exception as e:
            return f"Error generating answer: {str(e)}"
    
    def extract_timestamps(self, answer: str) -> List[float]:
        """Extract timestamps from LLM answer for potential video navigation"""
        patterns = [
            r'(\d+\.\d+)s',  # 123.45s
            r'(\d+)s',       # 123s
        ]
        
        timestamps = []
        for pattern in patterns:
            matches = re.findall(pattern, answer)
            for match in matches:
                timestamps.append(float(match))
        
        return sorted(set(timestamps))