#!/bin/bash

# Video RAG System Setup Script

echo "Setting up Video RAG System..."

# Create virtual environment
echo "Creating virtual environment..."
python3 -m venv venv
source venv/bin/activate

# Install dependencies
echo "Installing Python dependencies..."
pip install -r requirements.txt

# Download spaCy model
echo "Downloading spaCy model..."
python -m spacy download en_core_web_sm

# Create data directories
echo "Creating data directories..."
mkdir -p data/videos data/transcripts data/indexes

# Set up environment variables
if [ ! -f .env ]; then
    echo "Creating .env file..."
    echo "GROQ_API_KEY=your_groq_api_key_here" > .env
    echo "Please update .env with your actual Groq API key"
fi

echo "Setup complete! Don't forget to:"
echo "1. Update .env with your Groq API key"
echo "2. Activate virtual environment: source venv/bin/activate"
echo "3. Run the backend: uvicorn app.backend.main:app --reload"
echo "4. Run the frontend: streamlit run app/frontend/app.py"