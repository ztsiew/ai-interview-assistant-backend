# ia_backend.py
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import threading
import sounddevice as sd
import numpy as np
import openai
import wave
import tempfile
import os
import json
import uvicorn
import time
import io

# --- New Imports for Simulation ---
try:
    from pydub import AudioSegment
except ImportError:
    print("⚠️ pydub not found. Run: pip install pydub")

# ---------------------------------------------------------
# USER CONFIGURATION (EDIT THIS SECTION)
# ---------------------------------------------------------

# Set to True to use the simulation file instead of Microphone
USE_SIMULATION = True 

# Path to your test interview file (m4a, mp3, wav all work)
# Use 'r' before the string to handle Windows backslashes
SIMULATION_FILE_PATH = r"C:\Users\ZhiTao\OneDrive - Monash University\MAI Research\Python\MAIR - Session 3 - trimmed.m4a"

# Speed of simulation (0 = instant, 1 = real-time wait between chunks)
SIMULATION_DELAY = 1 

# ---------------------------------------------------------
# APP SETUP
# ---------------------------------------------------------
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

try:
    client = openai.OpenAI()
except:
    print("⚠️ OpenAI API Key missing.")

SAMPLE_RATE = 16000
BATCH_SECS = 15
WINDOW_SIZE = 8

# ---------------------------------------------------------
# GLOBAL STATE
# ---------------------------------------------------------
class InterviewSession:
    def __init__(self):
        self.thread = None
        self.stop_flag = threading.Event()
        self.transcript_history = []
        self.interview_plan = {}
        self.custom_prompt = """You are a qualitative research assistant. The interview transcripts does not have speaker labels and punctuation.
                            You should still interpret the dialogue correctly by inferring interviewer and interviewee turns internally"""
        self.latest_transcript = ""
        self.analysis_followup = "Waiting for interview..."
        self.analysis_transition = "Waiting for interview..."
        self.analysis_empathy = "Status: Normal"
        self.is_recording = False

session = InterviewSession()

# ---------------------------------------------------------
# CORE LOGIC (AI & TRANSCRIPTION)
# ---------------------------------------------------------
def save_wav(audio, filename):
    with wave.open(filename, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(audio.tobytes())

def transcribe_buffer(file_obj):
    """Helper to transcribe a file object directly"""
    try:
        res = client.audio.transcriptions.create(model="whisper-1", file=file_obj, language="en")
        return res.text
    except Exception as e:
        print(f"Transcription Error: {e}")
        return ""

def transcribe_numpy(audio_array):
    """Helper to transcribe a numpy array (from Mic)"""
    if audio_array.size == 0: return ""
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    tmp.close()
    try:
        save_wav(audio_array, tmp.name)
        with open(tmp.name, "rb") as f:
            return transcribe_buffer(f)
    except: return ""
    finally:
        if os.path.exists(tmp.name):
            try: os.remove(tmp.name)
            except: pass

def analyze_chunk(context, transcript, sys_prompt):
    if not transcript.strip(): return "Waiting..."
    messages = [
        {"role": "system", "content": f"{sys_prompt}\n\nContext:\n{context}"},
        {"role": "user", "content": f"Transcript:\n{transcript}"}
    ]
    try:
        res = client.chat.completions.create(model="gpt-4o-mini", messages=messages, max_tokens=250)
        output = res.choices[0].message.content

        # 🔍 DEBUG LOGGING (raw, frontend-independent)
        print("\n===== OPENAI RAW OUTPUT (repr) =====")
        print(repr(output))
        print("====================================\n")

        return output
    
    except Exception as e: return f"Error: {e}"


def generate_scorecard(full_text, plan):
    plan_str = json.dumps(plan, indent=2)
    prompt = f"""
    Analyze this interview based on plan: {plan_str}
    Transcript: {full_text}
    Provide a Markdown Scorecard (0-10) for:
    1. Topic Coverage 2. Clarity 3. Depth 4. Repetition 5. Bias 6. Prompt Adherence.
    """
    try:
        res = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "system", "content": "You are a strict coach."}, {"role": "user", "content": prompt}]
        )
        return res.choices[0].message.content
    except: return "Scorecard Error."

def run_ai_analysis(recent_text):
    """Shared AI logic for both Simulation and Live modes"""
    plan_str = json.dumps(session.interview_plan)
    
    # 1. Follow-up
    follow_up_prompt = f"""Identify the active topic in this Interview Plan: {plan_str}
Based on the transcript, suggest 2 probes. 
Instructions:
Identify Theme: Determine the active theme from the Interview Plan based on the most recent transcript segments.
Current Scope Only: You are strictly forbidden from suggesting questions that relate to future themes or topics in the plan that have not been reached yet. Focus entirely on "drilling down" into the current topic.
Deepen, Don't Shift: Provide "probes" that ask for more detail, emotional impact, or specific examples of what was just said.
Variation: Use different probing styles (e.g., one for clarification and one for elaboration). Avoid generic phrases like "Can you tell me more." Instead, use keywords from the transcript to make the questions context-specific.
Format: Exactly 2 questions, separated by a blank line, no other text.
    STRICT OUTPUT FORMAT:
    Question 1: [First deepening question] 
    [\n]
    Question 2: [Second deepening question]
    
    (Note: Do not include any other text.)
"""
    session.analysis_followup = analyze_chunk(
        follow_up_prompt,
        recent_text, session.custom_prompt
    )
    
    # 2. Transition
    transition_prompt = f"""
    You are an expert qualitative research assistant. 
    Using the Interview Plan below, perform these steps:
    1. **Identify State**: Determine which theme (using the exact name from the plan) is currently being discussed.
    2. **Target Next**: Find the immediate next title and its first question (question id ends with "_q1").
    3. **Draft Transition**: Write a conversational bridge (under 40 words). 
       Include: A brief acknowledgement of the current topic + a bridge to the next + the first question of the next theme.

    INTERVIEW PLAN:
    {plan_str}

    STRICT OUTPUT FORMAT:
    Current topic: [Exact Theme Name from Plan] 
    [\n]
    Transition: [Your conversational bridge and first question]
    
    (Note: Do not include any other text.)
    """
    session.analysis_transition = analyze_chunk(
        transition_prompt,
        recent_text, f"""{session.custom_prompt} and You are a professional research moderator. Avoid sounding robotic.""",
    )
    
    # 3. Empathy
    session.analysis_empathy = analyze_chunk(
        f"""Determine emotional distress/anxiety. If None, say: 'Status: Normal'.
        If Present: brief empathetic response (under 2 sentences).""",
        recent_text, "You are an emotional support classifier."
    )

# ---------------------------------------------------------
# SIMULATION LOOP (FILE BASED)
# ---------------------------------------------------------
def simulation_loop(file_path):
    print(f"🧪 Starting Simulation with file: {file_path}")
    
    if not os.path.exists(file_path):
        print("❌ File not found! Check SIMULATION_FILE_PATH")
        session.is_recording = False
        return

    try:
        # Load Audio
        audio = AudioSegment.from_file(file_path)
        chunk_length_ms = BATCH_SECS * 1000
        
        # Iterate through chunks
        for i in range(0, len(audio), chunk_length_ms):
            if session.stop_flag.is_set():
                break

            # 1. Slice audio
            chunk = audio[i : i + chunk_length_ms]
            
            # 2. Export to buffer (mocking a file object)
            buffer = io.BytesIO()
            buffer.name = "chunk.wav" # Whisper needs a filename attribute
            chunk.export(buffer, format="wav")
            buffer.seek(0) # Rewind to start of buffer

            # 3. Transcribe
            text = transcribe_buffer(buffer)
            
            if text.strip():
                session.transcript_history.append(text)
                recent = " ".join(session.transcript_history[-WINDOW_SIZE:])
                session.latest_transcript = recent
                
                # 4. Run Analysis
                print(f"Processing chunk {i//1000}s...")
                run_ai_analysis(recent)
            
            # 5. Wait (Simulate real time)
            if SIMULATION_DELAY > 0:
                time.sleep(SIMULATION_DELAY)

    except Exception as e:
        print(f"Simulation Crash: {e}")
    
    session.is_recording = False
    print("🧪 Simulation Ended.")

# ---------------------------------------------------------
# LIVE LOOP (MICROPHONE BASED)
# ---------------------------------------------------------
def live_loop():
    def callback(indata, frames, time, status):
        if session.stop_flag.is_set(): raise sd.CallbackStop()
        
        # Convert audio to integer format for wav saving
        text = transcribe_numpy((indata.copy() * 32767).astype(np.int16))
        
        if text.strip():
            session.transcript_history.append(text)
            recent = " ".join(session.transcript_history[-WINDOW_SIZE:])
            session.latest_transcript = recent
            run_ai_analysis(recent)

    with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, callback=callback, blocksize=int(SAMPLE_RATE*BATCH_SECS)):
        while not session.stop_flag.is_set():
            sd.sleep(100)

# ---------------------------------------------------------
# ENDPOINTS
# ---------------------------------------------------------
class ConfigUpdate(BaseModel):
    custom_prompt: str

@app.post("/upload_plan")
async def upload_plan(file: UploadFile = File(...)):
    try:
        content = await file.read()
        session.interview_plan = json.loads(content)
        return {"message": "Plan loaded"}
    except: return {"message": "Error loading JSON"}

@app.post("/update_config")
def update_config(config: ConfigUpdate):
    session.custom_prompt = config.custom_prompt
    return {"message": "Config updated"}

@app.post("/start")
def start_recording():
    if session.is_recording: return {"message": "Already running"}
    
    session.stop_flag.clear()
    session.transcript_history = []
    
    if USE_SIMULATION:
        # Run Simulation Mode
        session.thread = threading.Thread(target=simulation_loop, args=(SIMULATION_FILE_PATH,), daemon=True)
        message = "Simulation Started"
    else:
        # Run Live Mic Mode
        session.thread = threading.Thread(target=live_loop, daemon=True)
        message = "Live Recording Started"
        
    session.thread.start()
    session.is_recording = True
    return {"message": message}

@app.post("/stop")
def stop_recording():
    if not session.is_recording: return {"message": "Not running", "scorecard": ""}
    
    session.stop_flag.set()
    if session.thread:
        session.thread.join()
    session.is_recording = False
    
    full_text = " ".join(session.transcript_history)
    # score = generate_scorecard(full_text, session.interview_plan)
    return {"message": "Stopped"
            # , "scorecard": score
            }

@app.get("/status")
def get_status():
    # Return the last WINDOW_SIZE segments as a list instead of a joined string
    recent_segments = session.transcript_history[-WINDOW_SIZE:] if session.transcript_history else []
    
    return {
        "is_recording": session.is_recording,
        "transcript_list": recent_segments, # Change this to a list
        "followup": session.analysis_followup,
        "transition": session.analysis_transition,
        "empathy": session.analysis_empathy
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)