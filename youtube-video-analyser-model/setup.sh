#!/bin/bash

# Setup script for YouTube Video Analyzer Model
echo "🚀 Setting up YouTube Video Analyzer Model..."

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is required but not installed."
    exit 1
fi

echo "✓ Python 3 found"

# Check if Poetry is available, otherwise use pip
if command -v poetry &> /dev/null; then
    echo "✓ Poetry found, using Poetry for dependency management"
    USE_POETRY=true
else
    echo "⚠️  Poetry not found, using pip"
    USE_POETRY=false
fi

# Create virtual environment if not using Poetry
if [ "$USE_POETRY" = false ]; then
    if [ ! -d "venv" ]; then
        echo "📦 Creating virtual environment..."
        python3 -m venv venv
    fi
    
    echo "🔧 Activating virtual environment..."
    source venv/bin/activate
fi

# Install dependencies
echo "📦 Installing dependencies..."
if [ "$USE_POETRY" = true ]; then
    poetry install
else
    pip install -r requirements.txt
fi

# Download spaCy model
echo "🔧 Setting up spaCy language model..."
if [ "$USE_POETRY" = true ]; then
    poetry run python -m spacy download en_core_web_sm
else
    python -m spacy download en_core_web_sm
fi

# Create .env file from example
if [ ! -f ".env" ]; then
    echo "⚙️  Creating .env file from example..."
    cp .env.example .env
    echo "📝 Please edit .env file to add your API keys"
else
    echo "✓ .env file already exists"
fi

# Create data directory if not exists
mkdir -p data

# Run installation test
echo "🧪 Running installation test..."
if [ "$USE_POETRY" = true ]; then
    poetry run python test_installation.py
else
    python test_installation.py
fi

if [ $? -eq 0 ]; then
    echo ""
    echo "🎉 Setup completed successfully!"
    echo ""
    echo "Next steps:"
    echo "1. Edit the .env file to add your API keys:"
    echo "   - GROQ_API_KEY for Groq (recommended)"
    echo "   - OPENAI_API_KEY for OpenAI"
    echo "   - OPENROUTER_API_KEY for OpenRouter"
    echo ""
    echo "2. Test with a YouTube video:"
    if [ "$USE_POETRY" = true ]; then
        echo "   poetry run python examples/basic_usage.py"
        echo "   # OR"
        echo "   poetry run youtube-analyzer analyze 'https://www.youtube.com/watch?v=VIDEO_ID'"
    else
        echo "   python examples/basic_usage.py"
        echo "   # OR"
        echo "   python -m youtube_analyzer.cli analyze 'https://www.youtube.com/watch?v=VIDEO_ID'"
    fi
    echo ""
    echo "3. See examples/ directory for more usage examples"
    echo ""
    echo "📚 Read the README.md for detailed documentation"
else
    echo "❌ Setup failed. Please check the errors above."
    exit 1
fi