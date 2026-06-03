import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled, NoTranscriptFound
import re

# --- 1. Functions from the Notebook ---

def extract_video_id(url):
    """Extracts video ID from different YouTube URL formats."""
    match = re.search(r"(?:v=|youtu\.be/)([a-zA-Z0-9_-]{11})", url)
    return match.group(1) if match else None

def get_transcript(video_id):
    """Fetch transcript using the NEW API format."""
    try:
        api = YouTubeTranscriptApi()
        transcript = api.fetch(video_id)
        return " ".join([t.text for t in transcript])
    except TranscriptsDisabled:
        return "Error: Transcripts are disabled for this video."
    except NoTranscriptFound:
        return "Error: No transcript found for this video."
    except Exception as e:
        return f"Error: {str(e)}"

def chunk_text(text, chunk_size=1200):
    sentences = text.split(". ")
    chunks, current_chunk = [], ""

    for sentence in sentences:
        if len(current_chunk) + len(sentence) < chunk_size:
            current_chunk += sentence + ". "
        else:
            chunks.append(current_chunk.strip())
            current_chunk = sentence + ". "

    if current_chunk:
        chunks.append(current_chunk.strip())
    return chunks

def summarize_chunk(text_chunk):
    # We give the model a specific instruction (prompt engineering)
    prompt = f"Summarize the following text clearly:\n{text_chunk}"

    # Convert text to tensor numbers (inputs)
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=1024
    ).to(device)

    # Generate the summary
    summary_ids = model.generate(
        **inputs,
        max_new_tokens=120,
        num_beams=4,
        length_penalty=1.0,
        early_stopping=True
    )

    # Decode back to text
    return tokenizer.decode(summary_ids[0], skip_special_tokens=True)

# --- 2. Model Loading (cached) ---

@st.cache_resource
def load_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_name = "google/flan-t5-base"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)
    return tokenizer, model, device

tokenizer, model, device = load_model()

# --- 3. Streamlit UI ---

st.set_page_config(page_title="YouTube Video Summarizer", layout="wide")
st.title("🎬 YouTube Video Summarizer")
st.markdown("Enter a YouTube video URL below to get AI-generated notes.")

video_url = st.text_input("YouTube Video URL", placeholder="e.g., https://www.youtube.com/watch?v=dQw4w9WgXcQ")

if video_url:
    video_id = extract_video_id(video_url)

    if not video_id:
        st.error("Invalid YouTube URL. Please provide a valid link.")
    else:
        st.video(video_url) # Display the video
        st.subheader("Generating Notes...")

        with st.spinner("Fetching transcript..."):
            transcript = get_transcript(video_id)

        if transcript.startswith("Error"):
            st.error(transcript)
        else:
            st.success("Transcript fetched successfully!")
            st.text_area("Full Transcript (first 1000 characters)", transcript[:1000] + '...' if len(transcript) > 1000 else transcript, height=150, disabled=True)

            with st.spinner("Chunking transcript and summarizing..."):
                chunks = chunk_text(transcript)
                notes = []
                progress_bar = st.progress(0)

                for i, chunk in enumerate(chunks):
                    summary = summarize_chunk(chunk)
                    notes.append(f"- {summary}")
                    progress_bar.progress((i + 1) / len(chunks))
            st.success("Summarization complete!")

            st.subheader("📝 AI GENERATED NOTES")
            for note in notes:
                st.markdown(note)
