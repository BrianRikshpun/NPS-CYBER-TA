# NPS Cyber TA

This project is an AI-powered Teaching Assistant built with **Gradio**, **OpenAI**, and **Ollama backend**.
It provides chat, image generation, and speech interaction capabilities.

## Features
- 💬 Chatbot with memory (OpenAI/Ollama backend)
- 🖼️ Image generation with DALL·E 3 (fallback if org is verified)
- 🎤 Text-to-speech voice responses
- 🎛️ Web UI built using Gradio

## Installation

```bash
git clone <your-repo-url>
cd NPS-CYBER-TA
python -m venv .venv
source .venv/bin/activate   # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Running the App

```bash
python app.py
```

Then open the link shown in terminal (default: `http://127.0.0.1:7860`).

## Environment Variables

Set your OpenAI API Key before running:

**Linux / macOS**
```bash
export OPENAI_API_KEY="your_api_key"
```

**Windows (PowerShell)**
```powershell
setx OPENAI_API_KEY "your_api_key"
```

## Notes
- Ensure Ollama is installed and models are pulled locally if you want to use it as backend.
- Image generation may require verified OpenAI organization.
