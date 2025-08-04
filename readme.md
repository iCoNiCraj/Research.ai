## Research.AI

AI-powered research paper search and audio assistant.

---

## Overview

**Research.AI** enables users to search for research papers and generate real-life voice audio summaries from paper URLs.  
The project is designed to be modular and extensible, supporting multiple search sources and voice synthesis backends.

---

## Voice Synthesis: Current Status

- **Currently Used:**  
  We use **Google Text-to-Speech (gTTS)** for generating audio summaries. This is because gTTS is lightweight and works well on laptops or low-resource servers.
- **Preferred (Not Yet Default):**  
  **HuggingFace Kokoro** and **Dia** provide much higher quality, human-like voice synthesis. However, they require significant GPU resources and are not practical to run on most personal laptops or basic servers.
- **How to Upgrade:**  
  If you have access to a powerful GPU server, you can swap out gTTS for HuggingFace Kokoro/Dia in the backend for much more realistic audio.

---

## How it Works

1. **User Input:**  
   Enter a search term or paste a research paper URL in the search bar.

2. **Paper Search:**

   - If a keyword is entered, the frontend queries the backend (`/query?q=...`) to fetch relevant research papers from multiple sources (arXiv, Semantic Scholar, etc.).
   - Results are displayed in a list.

3. **Audio Generation:**

   - For any paper (from search or direct URL), click "Generate Audio".
   - The frontend sends a request to the backend (`/create_podcast?url=...`).
   - The backend extracts and summarizes the paper content.
   - The backend uses **gTTS** (or HuggingFace models if enabled) to generate an audio summary.
   - The generated audio file is returned to the frontend.

4. **Playback & Download:**
   - The frontend displays an audio player for the generated summary.
   - Users can listen to or download the audio file.

---

## Frontend

**Tech Stack:** Next.js, React, TailwindCSS, Zustand (state), Radix UI.

**Key Components:**

- **SearchBar:**  
  Lets users search for papers by keyword or enter a URL for audio generation. Handles tab switching between search and audio generation modes.

- **SearchResults:**  
  Displays search results from all sources. Allows users to copy paper URLs or generate audio summaries.

- **AudioPlayer:**  
  Plays generated audio and provides download functionality.

- **State Management:**  
  Uses Zustand (`/lib/store.ts`) to manage search results, loading states, and the current podcast URL.

- **UI Components:**  
  Modular UI built with Radix UI and TailwindCSS for a modern, responsive experience.

---

## Backend (Design & File Responsibilities)

> **Note:** Backend code is not included in this workspace, but here's how it is structured and how each file/agent works.

### Main Endpoints

- `GET /query?q=...`  
  Searches for research papers using multiple sources and LLMs.
- `GET /create_podcast?url=...`  
  Generates an audio summary from a paper URL.

---

## Example Backend Flow

1. **Receive a paper URL or search query.**
2. **Search:**
   - `search.py` calls `arxiv.py` and `semantic_scholar.py` to fetch papers.
   - Results are merged and returned.
3. **Summarize:**
   - `llm.py` or `agentic_llm.py` summarizes the paper.
   - If agentic, the LLM can call tools (e.g., arXiv search) as needed.
4. **Synthesize Audio:**
   - `tts.py` generates audio using gTTS (or HuggingFace if enabled).
5. **Return:**
   - The audio file is sent to the frontend for playback/download.

---

## Getting Started

### Prerequisites

- Node.js (v18+)
- Backend server (Python/Node, with endpoints as above, and access to gTTS or HuggingFace models)

### Frontend Setup

1. Clone this repo.
2. Set the backend URL in your environment variables (`NEXT_PUBLIC_BACKEND_URL`).
3. Install dependencies and run the frontend.

### Backend Setup

1. Implement `/query` and `/create_podcast` endpoints.
2. Integrate arXiv, Semantic Scholar, and LLM summarization.
3. Use gTTS for TTS by default; switch to HuggingFace Kokoro/Dia if you have a GPU server.
4. Example libraries: `transformers`, `gtts`, `flask`/`fastapi` (Python).

---

## Usage

1. Search for papers or enter a URL.
2. Click "Generate Audio" to create and play/download the audio summary.

---

## Major Improvements

- Modular backend: Easily add new sources or LLMs.
- Real-life voice synthesis with HuggingFace Kokoro/Dia (if hardware allows).
- Fast local TTS with gTTS for instant feedback.
- Modern, user-friendly UI.

---

## File-by-File Backend Explanation

- **app.py:**  
  Main backend server. Exposes the API endpoints (`/query`, `/create_podcast`) for searching papers and generating audio summaries. Handles request routing and integrates all backend modules.

- **audio.py:**  
  Handles text-to-speech (TTS) synthesis. Uses gTTS by default for fast, local audio generation. Can be extended to use HuggingFace Kokoro/Dia for high-quality, real-life voice if hardware allows.

- **llm.py:**  
  Contains logic for summarizing research papers using a language model (LLM). Supports different LLM backends (OpenAI, HuggingFace, or local models). Responsible for generating concise summaries from paper content.

- **tools.py:**  
  Utility functions and helper methods for the backend. May include functions for formatting, error handling, deduplication, or integration with external APIs.

- **archive/agent.py:**  
  (If used) Implements agentic workflows, where the LLM can autonomously decide to call tools (e.g., search, summarization) as part of its reasoning process.

- **audio_cache/**  
  Stores generated audio files (WAV format) for quick retrieval and to avoid redundant synthesis.

---

_Let me know if you want backend code samples, more details, or help with deployment!_

-
