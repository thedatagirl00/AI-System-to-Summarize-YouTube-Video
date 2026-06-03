# AI-System-to-Summarize-YouTube-Video
# YouTube Video Summarizer with Streamlit and Hugging Face Transformers
# Overview
This project provides a Streamlit web application that allows users to generate AI-powered summaries of YouTube videos. It leverages the `youtube-transcript-api` to fetch video transcripts and the `transformers` library (specifically a Flan-T5 model) for text summarization.

## Features

*   **Video ID Extraction**: Automatically extracts the YouTube video ID from various URL formats.
*   **Transcript Fetching**: Retrieves the full transcript of a given YouTube video.
*   **Smart Chunking**: Divides long transcripts into manageable chunks for efficient summarization.
*   **AI-Powered Summarization**: Utilizes a pre-trained sequence-to-sequence model (Flan-T5) to generate concise summaries for each transcript chunk.
*   **Interactive UI**: A user-friendly Streamlit interface for inputting video URLs and displaying notes.

## Technologies Used

*   Python 3.x
*   Streamlit
*   `youtube-transcript-api`
*   `transformers` (Hugging Face)
*   `accelerate`
*   `sentencepiece`
*   `torch`

## Setup and Installation

To run this project locally, follow these steps:

1.  **Clone the repository**:

    ```bash
    git clone <your-repository-url>
    cd <your-repository-name>
    ```

2.  **Create a virtual environment** (recommended):

    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install dependencies**:
    The `requirements.txt` file lists all the necessary Python packages. Ensure you have these installed:

    ```bash
    pip install -r requirements.txt
    ```

    The `requirements.txt` for this project should contain:

    ```
    streamlit
    youtube-transcript-api
    transformers
    accelerate
    sentencepiece
    torch
    ```

## Running the Streamlit Application

1.  **Ensure `app.py` exists**: The main application logic is in `app.py`. If you've been working in a Colab notebook, make sure you've saved the content of the `%%writefile app.py` cell to a local file named `app.py`.

2.  **Run the app**:

    ```bash
    streamlit run app.py
    ```

    This will open the Streamlit application in your default web browser (usually at `http://localhost:8501`).

## Deployment to Hugging Face Spaces

This application is designed to be easily deployable to [Hugging Face Spaces](https://huggingface.co/spaces).

1.  **Create a New Space**: Go to Hugging Face Spaces, click "Create new Space", choose "Streamlit" as the SDK, and give your Space a name.

2.  **Upload Files**: Upload your `app.py` and `requirements.txt` files to the Space. Hugging Face will automatically detect these files and build your application.

    *   **`app.py`**: Contains the Streamlit UI and the core logic (video ID extraction, transcript fetching, chunking, summarization).
    *   **`requirements.txt`**: Lists all Python dependencies (`streamlit`, `youtube-transcript-api`, `transformers`, `accelerate`, `sentencepiece`, `torch`).

Once deployed, your Streamlit app will be live and accessible via the Space's URL.

## How to Use

1.  Paste a YouTube video URL into the input field.
2.  Click "Generate Notes" (or simply press Enter after pasting the URL, as Streamlit apps often re-run on input changes).
3.  The application will display the video, fetch its transcript, chunk it, summarize each chunk using AI, and present the generated notes.

Enjoy summarizing your YouTube videos!
```
