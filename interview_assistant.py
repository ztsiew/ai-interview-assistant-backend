import gradio as gr
import openai
import numpy as np
import tempfile
import os
import wave
import sounddevice as sd
import threading
import json
import time

# -----------------------------
# Config & Initialization
# -----------------------------
try:
    client = openai.OpenAI()  # Assumes OPENAI_API_KEY is in env variables
except openai.OpenAIError:
    print("OpenAI API key not found. Please set the OPENAI_API_KEY environment variable.")
    exit()

SAMPLE_RATE = 16000
BATCH_SECS = 15  # Slightly faster updates
MAX_TOKENS = 600
WINDOW_SIZE = 8  # Keep slightly more context for the LLM

# -----------------------------
# Backend Logic
# -----------------------------

def save_wav(audio, filename, samplerate=SAMPLE_RATE):
    """Saves numpy audio array to WAV."""
    with wave.open(filename, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(samplerate)
        wf.writeframes(audio.tobytes())

def transcribe(audio):
    """Transcribes audio using OpenAI Whisper."""
    if audio.size == 0:
        return ""
    tmpfile = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    tmpfile_path = tmpfile.name
    tmpfile.close()
    save_wav(audio, tmpfile_path)

    try:
        with open(tmpfile_path, "rb") as f:
            transcript = client.audio.transcriptions.create(
                model="whisper-1",
                file=f,
                language="en"
            )
        return transcript.text
    except Exception as e:
        return f"[Transcription Error: {e}]"
    finally:
        if os.path.exists(tmpfile_path):
            os.remove(tmpfile_path)

def analyze_chunk(context_prompt, transcript_text, custom_system_prompt):
    """Generic helper to send a prompt to GPT-4o-mini."""
    if not transcript_text.strip():
        return "Waiting for speech..."
    
    # Combine user's custom prompt with the specific task prompt
    messages = [
        {"role": "system", "content": f"{custom_system_prompt}\n\nContext:\n{context_prompt}"},
        {"role": "user", "content": f"Current Transcript segment:\n{transcript_text}"}
    ]
    
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            max_tokens=250,
            temperature=0.7
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Analysis Error: {e}"

def generate_final_scorecard(full_transcript, interview_plan):
    """Generates the post-interview scorecard using GPT-4o."""
    if not full_transcript:
        return "No transcript recorded."
    
    plan_str = json.dumps(interview_plan, indent=2) if interview_plan else "No specific plan uploaded."
    
    prompt = f"""
    You are an expert interview coach. Analyze the following full interview transcript against the provided plan.
    
    Interview Plan:
    {plan_str}
    
    Full Transcript:
    {full_transcript}
    
    Provide a Scorecard with these 6 criteria (score X/10 for each):
    1. Topic Coverage (Did they stick to the plan?)
    2. Clarity of Questions
    3. Depth of Follow-up
    4. Repetition Level (Lower repetition = higher score)
    5. Bias Awareness
    6. Prompt Integration (Did they follow custom instructions?)
    
    Also provide:
    - Key Insights
    - Recommendations for Future Sessions
    
    Format the output nicely using Markdown.
    """
    
    try:
        response = client.chat.completions.create(
            model="gpt-4o", # Using bigger model for final analysis
            messages=[{"role": "system", "content": "You are a critical but constructive interview coach."},
                      {"role": "user", "content": prompt}],
            temperature=0.7
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Scorecard Generation Error: {e}"

# -----------------------------
# Live Loop (Background Thread)
# -----------------------------
def live_loop(state_dict):
    """
    Main loop: records audio, appends to transcript, and runs 3 parallel analysis tasks.
    """
    def audio_callback(indata, frames, time, status):
        if state_dict["stop_flag"].is_set():
            raise sd.CallbackStop()
        
        audio_chunk = indata.copy()
        text_chunk = transcribe((audio_chunk * 32767).astype(np.int16))
        
        if text_chunk.strip():
            # Update transcript history
            state_dict["transcript_history"].append(text_chunk)
            
            # Keep a rolling window for live analysis
            recent_window = state_dict["transcript_history"][-WINDOW_SIZE:]
            recent_text = " ".join(recent_window)
            
            # --- Prepare Contexts ---
            plan_str = json.dumps(state_dict.get("interview_plan", {}), indent=2)
            user_custom_prompt = state_dict.get("custom_prompt", "")
            
            # --- 1. Follow-up Questions ---
            p_followup = f"""
            Based on the Interview Plan below, suggest 2 relevant follow-up questions for the current topic being discussed.
            Avoid jumping to future topics in the plan yet.
            Plan: {plan_str}
            """
            state_dict["analysis_followup"] = analyze_chunk(p_followup, recent_text, user_custom_prompt)
            
            # --- 2. Transition Helper ---
            p_transition = f"""
            Identify which topic from the Interview Plan we are currently on. 
            Suggest a smooth transition sentence to move to the *next* logical topic in the plan.
            Plan: {plan_str}
            """
            state_dict["analysis_transition"] = analyze_chunk(p_transition, recent_text, user_custom_prompt)
            
            # --- 3. Empathy Cues ---
            p_empathy = """
            Analyze the tone and content. If the interviewee seems emotional, nervous, or stuck, provide 
            short, supportive empathy cues (e.g., "Take your time," "That sounds challenging"). 
            If the conversation is normal/factual, reply with "Status: Normal".
            """
            state_dict["analysis_empathy"] = analyze_chunk(p_empathy, recent_text, "Be supportive and empathetic.")
            
            # Update latest transcript for UI
            state_dict["latest_transcript_text"] = recent_text

    # Start Recording Stream
    try:
        with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype='float32',
                            blocksize=int(SAMPLE_RATE * BATCH_SECS),
                            callback=audio_callback):
            while not state_dict["stop_flag"].is_set():
                sd.sleep(100)
    except Exception as e:
        print(f"Stream Error: {e}")

# -----------------------------
# UI Handlers
# -----------------------------

def process_json_upload(file_obj, state):
    if file_obj is None:
        return "No file uploaded.", state
    try:
        with open(file_obj.name, 'r') as f:
            data = json.load(f)
        state["interview_plan"] = data
        return "✅ Plan loaded successfully!", state
    except Exception as e:
        return f"❌ Error loading JSON: {e}", state

def start_recording(state):
    if state["thread"] is None:
        state["stop_flag"] = threading.Event()
        state["transcript_history"] = [] # Clear history on new start
        state["thread"] = threading.Thread(target=live_loop, args=(state,), daemon=True)
        state["thread"].start()
        return "🔴 Recording...", "⏹ Stop Recording", state
    return "🔴 Recording...", "⏹ Stop Recording", state

def stop_recording(state):
    if state["thread"] is not None:
        state["stop_flag"].set()
        state["thread"].join()
        state["thread"] = None
        state["stop_flag"] = None
        
        # Auto-generate scorecard
        full_text = " ".join(state["transcript_history"])
        scorecard = generate_final_scorecard(full_text, state.get("interview_plan"))
        
        return "Stopped.", "▶ Start Recording", scorecard, state
    return "Stopped.", "▶ Start Recording", "", state

def update_dashboard(state):
    return (
        state["latest_transcript_text"],
        state["analysis_followup"],
        state["analysis_transition"],
        state["analysis_empathy"]
    )

# -----------------------------
# Gradio UI Layout
# -----------------------------
custom_css = """
.gradio-container { font-family: 'Helvetica', 'Arial', sans-serif; }
.section-header { color: #2c3e50; font-weight: bold; margin-bottom: 10px; }
.gr-button.primary { background-color: #3498db; color: white; }
.gr-button.stop { background-color: #e74c3c; color: white; }
.scorecard-box { background-color: #f9f9f9; border: 1px solid #ddd; padding: 15px; border-radius: 5px; }
"""

with gr.Blocks(theme=gr.themes.Soft(), css=custom_css, title="AI Interview Assistant") as demo:
    
    # --- Shared State ---
    app_state = gr.State({
        "thread": None,
        "stop_flag": None,
        "transcript_history": [],
        "interview_plan": {}, # Stores the JSON
        "custom_prompt": "You are a helpful assistant.",
        "latest_transcript_text": "",
        "analysis_followup": "Waiting for data...",
        "analysis_transition": "Waiting for data...",
        "analysis_empathy": "Status: Normal"
    })

    gr.Markdown("# 🎤 AI Interview Co-Pilot")
    
    # --- Phase 1: Setup ---
    with gr.Row(variant="panel"):
        with gr.Column(scale=1):
            gr.Markdown("### 1. Upload Interview Plan (JSON)")
            json_file = gr.File(label="Upload Plan", file_types=[".json"])
            upload_status = gr.Textbox(label="Status", value="Waiting for upload...", interactive=False)
            json_file.upload(process_json_upload, inputs=[json_file, app_state], outputs=[upload_status, app_state])
            
        with gr.Column(scale=1):
            gr.Markdown("### 2. Custom Pre-Prompt")
            custom_prompt_input = gr.Textbox(
                label="Instructions for AI (e.g., 'Focus on social impact')", 
                value="Keep suggestions concise. Focus on strategic depth.",
                lines=3
            )
            # Update state immediately when changed
            custom_prompt_input.change(
                lambda val, st: (st.update({"custom_prompt": val}), st)[1],
                inputs=[custom_prompt_input, app_state], outputs=[app_state]
            )

    gr.Markdown("---")

    # --- Phase 2: Live Dashboard ---
    with gr.Row():
        # Left: Controls & Transcript
        with gr.Column(scale=1):
            gr.Markdown("### 🔴 Live Controls")
            status_indicator = gr.Textbox(label="System Status", value="Stopped", interactive=False)
            toggle_btn = gr.Button("▶ Start Recording", variant="primary")
            
            gr.Markdown("### 📜 Live Transcript")
            transcript_box = gr.Textbox(label="Recent Context", lines=10, interactive=False)

        # Right: The 3 Pillars (Tabs)
        with gr.Column(scale=2):
            gr.Markdown("### 🧠 Real-time AI Insights")
            with gr.Tabs():
                with gr.TabItem("🔍 Follow-up Questions"):
                    followup_box = gr.Markdown("Waiting for interview to start...")
                with gr.TabItem("twisted_rightwards_arrows Transition Helper"):
                    transition_box = gr.Markdown("Waiting for interview to start...")
                with gr.TabItem("❤️ Empathy Cues"):
                    empathy_box = gr.Markdown("Waiting for interview to start...")

    gr.Markdown("---")

    # --- Phase 3: Post-Interview Scorecard ---
    with gr.Group(elem_classes="scorecard-box"):
        gr.Markdown("## 📊 Post-Interview Scorecard")
        scorecard_display = gr.Markdown("Scorecard will appear here after you stop recording.")

    # --- Event Wiring ---
    
    # Start/Stop Logic
    def toggle_handler(state):
        if state["thread"] is None:
            # Starting
            msg, btn_txt, st = start_recording(state)
            return msg, btn_txt, "Generating Scorecard...", st
        else:
            # Stopping
            msg, btn_txt, score, st = stop_recording(state)
            return msg, btn_txt, score, st

    toggle_btn.click(
        toggle_handler,
        inputs=[app_state],
        outputs=[status_indicator, toggle_btn, scorecard_display, app_state]
    )

    # Timer for UI Refresh (ticks every 1s)
    timer = gr.Timer(1.0)
    timer.tick(
        update_dashboard,
        inputs=[app_state],
        outputs=[transcript_box, followup_box, transition_box, empathy_box]
    )

if __name__ == "__main__":
    demo.launch()

