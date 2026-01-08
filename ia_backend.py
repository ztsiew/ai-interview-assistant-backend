# backend.py
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

# --- Config ---
app = FastAPI()

# !!! CRITICAL: Allow the browser to talk to this API !!!
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
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

# --- Global State ---
class InterviewSession:
    def __init__(self):
        self.thread = None
        self.stop_flag = None
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

# --- Audio & AI Logic (Same as before) ---
def save_wav(audio, filename):
    with wave.open(filename, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(audio.tobytes())

def transcribe(audio):
    if audio.size == 0: return ""
    # Windows Fix: Create -> Close -> Use
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    tmp.close()
    try:
        save_wav(audio, tmp.name)
        with open(tmp.name, "rb") as f:
            res = client.audio.transcriptions.create(model="whisper-1", file=f, language="en")
        return res.text
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
        return res.choices[0].message.content
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

# --- Background Loop ---
def live_loop():
    def callback(indata, frames, time, status):
        if session.stop_flag.is_set(): raise sd.CallbackStop()
        text = transcribe((indata.copy() * 32767).astype(np.int16))
        if text.strip():
            session.transcript_history.append(text)
            recent = " ".join(session.transcript_history[-WINDOW_SIZE:])
            session.latest_transcript = recent
            
            plan_str = json.dumps(session.interview_plan)
            session.analysis_followup = analyze_chunk(f"""Suggest 2 short follow-ups based on plan below. 
                                                      You must identify which topic / question are we at, you must not suggest any questions
                                                      listed in the upcoming interview topic. Your job is to probe deeper to gain more insights. \n{plan_str}""",
                                                        recent, session.custom_prompt)
            session.analysis_transition = analyze_chunk(f"""Identify current topic for the plan below and suggest transition statement (to 
                                                        help interviewer better transition to next topic on the plan with minimal awkwardness). 
                                                        Keep your response under 25 words excluding the next question.
                                                         \n{plan_str}""",
                                                          recent, session.custom_prompt)
            session.analysis_empathy = analyze_chunk(f"""Your task: Determine whether it contains emotional distress, frustration, anxiety or need for emotional support.
                                                        If NO emotional content is present, respond with exactly: Status: Normal \n
                                                    If emotional content IS present:
                                                    - Respond with a brief, calm, empathetic message.
                                                    - Acknowledge the emotion explicitly (e.g. frustration, stress, uncertainty).
                                                    - Do NOT over-intensify or dramatize.
                                                    - Keep the response under 2 sentences.""",
                                                      recent,
                                                     "You are an emotional support classifier and responder.")

    with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, callback=callback, blocksize=int(SAMPLE_RATE*BATCH_SECS)):
        while not session.stop_flag.is_set():
            sd.sleep(100)


# --- Endpoints ---
class ConfigUpdate(BaseModel):
    custom_prompt: str

@app.post("/upload_plan")
async def upload_plan(file: UploadFile = File(...)):
    try:
        content = await file.read()
        session.interview_plan = json.loads(content)
        return {"message": "Plan loaded"}
    except: return {"message": "Error"}

@app.post("/update_config")
def update_config(config: ConfigUpdate):
    session.custom_prompt = config.custom_prompt
    return {"message": "Updated"}

@app.post("/start")
def start_recording():
    if session.is_recording: return {"message": "Already recording"}
    session.stop_flag = threading.Event()
    session.transcript_history = []
    session.thread = threading.Thread(target=live_loop, daemon=True)
    session.thread.start()
    session.is_recording = True
    return {"message": "Started"}

@app.post("/stop")
def stop_recording():
    if not session.is_recording: return {"message": "Not recording", "scorecard": ""}
    session.stop_flag.set()
    session.thread.join()
    session.is_recording = False
    full_text = " ".join(session.transcript_history)
    score = generate_scorecard(full_text, session.interview_plan)
    return {"message": "Stopped", "scorecard": score}

@app.get("/status")
def get_status():
    return {
        "is_recording": session.is_recording,
        "transcript": session.latest_transcript,
        "followup": session.analysis_followup,
        "transition": session.analysis_transition,
        "empathy": session.analysis_empathy
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)