# ia_backend_simulation.py
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
from pypdf import PdfReader  # pip install pypdf

# ---------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------
USE_SIMULATION = True 
SIMULATION_FILE_PATH = r"C:\Users\ZhiTao\OneDrive - Monash University\MAI Research\Python\ai-interview-assistant-backend\MAIR - Session 3 - trimmed.m4a"
SIMULATION_DELAY = 1 
SAMPLE_RATE = 16000
WINDOW_SIZE_FOLLOW_UP = 8
WINDOW_SIZE_TRANSITION = 2
WINDOW_SIZE_EMPATHY = 4
BATCH_SECS = 15

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

# ---------------------------------------------------------
# GLOBAL STATE
# ---------------------------------------------------------
class InterviewSession:
    def __init__(self):
        self.thread = None
        self.stop_flag = threading.Event()
        self.transcript_history = []
        # Default empty structure until PDF is uploaded
        self.interview_plan = {"interview_guides_collection": []} 
        self.system_identity = (
            "You are a qualitative research assistant. The interview transcripts do "
            "not have speaker labels and punctuation. You should still interpret the "
            "dialogue correctly by inferring interviewer and interviewee turns internally."
        )
        self.custom_prompt = ""
        self.latest_transcript = ""
        self.analysis_followup = "Waiting for interview..."
        self.analysis_transition = "Waiting for interview..."
        self.analysis_empathy = "Status: Normal"
        self.is_recording = False

session = InterviewSession()

# ---------------------------------------------------------
# CORE LOGIC
# ---------------------------------------------------------
def transcribe_buffer(file_obj):
    try:
        res = client.audio.transcriptions.create(model="whisper-1", file=file_obj, language="en")
        return res.text
    except Exception as e:
        print(f"Transcription Error: {e}")
        return ""


def analyze_chunk(context, transcript, sys_prompt):
    if not transcript.strip(): 
        return "Waiting..."

    if sys_prompt.strip():
        priority_directive = f"### CRITICAL PRIORITY INSTRUCTIONS:\n{sys_prompt}\n\n---\n"
    else:
        priority_directive = ""

    system_content = (
        f"{priority_directive}"
        f"{session.system_identity}\n\n"
        f"INTERVIEW CONTEXT & GUIDELINES:\n{context}"
    )

    messages = [
        {"role": "system", "content": system_content},
        {"role": "user", "content": f"Transcript:\n{transcript}"}
    ]

    try:
        res = client.chat.completions.create(model="gpt-4o", messages=messages, max_tokens=250)
        output = res.choices[0].message.content

        # 🔍 DEBUG LOGGING
        print("\n===== OPENAI RAW OUTPUT (repr) =====")
        print(repr(output))
        print("====================================\n")

        return output
    except Exception as e: 
        return f"Error: {e}"


def run_ai_analysis():
    """Uses the PDF-generated plan to drive the interview logic with priority instructions."""
    plan_str = json.dumps(session.interview_plan)
    
    # 1. Follow-up (Deepen)
    recent_text_followup = " ".join(session.transcript_history[-WINDOW_SIZE_FOLLOW_UP:])

    follow_up_prompt = f"""Identify the active topic in this Interview Plan: {plan_str}
    Based on the transcript, suggest 2 probes. 
    Instructions:
    Identify Theme: Determine the active theme from the Interview Plan based on the most recent transcript segments.
    Current Scope Only: You are strictly forbidden from suggesting questions that relate to future themes or topics in the plan that have not been reached yet. Focus entirely on "drilling down" into the current topic.
    Deepen, Don't Shift: Provide "probes" that ask for more detail, emotional impact, or specific examples of what was just said.
    Variation: Use different probing styles and choose the most suitable ones (e.g.: clarification probe, elaboration probe, contrast probe, consequence probe, interpretive probe or echo probe).
    Use keywords from the transcript to make the questions context-specific.
    Format: Exactly 2 questions, separated by a blank line, no other text.

    Example Output:
    **Question 1:** [Follow-up question 1]

    **Question 2:** [Follow-up question 2]

    OUTPUT RULES:
    - Use exactly the format above.
    - Do NOT include any extra explanation or notes.
    - Do NOT ADD references question.
    - No square brackets [] in output.
    - **Note:** Prioritize any 'New Instructions' provided in the system context over these default instructions if they conflict.
    """

    session.analysis_followup = analyze_chunk(
        follow_up_prompt, recent_text_followup, session.custom_prompt
    )
    
    # 2. Transition (Shift)
    recent_text_transition = " ".join(session.transcript_history[-WINDOW_SIZE_TRANSITION:])

    transition_prompt = f"""
    Using this Plan: {plan_str}
    TASK:
    1. Identify the CURRENT theme title from the Interview Plan.
    2. Identify the IMMEDIATE NEXT theme title and its FIRST question only.
    3. Write ONE smooth conversational transition that bridges them.

    STRICT RULES:
    - Theme title = title only (max 6 words).
    - DO NOT include question lists, evidence, quotes from the plan, arrays, brackets, or JSON.
    - DO NOT explain your reasoning.
    - DO NOT output anything outside the format.
    - **Note:** Follow any 'New Instructions' provided in the system header regarding transition tone or style.

    STRICT OUTPUT FORMAT (Markdown only):

    **Current Topic:** "There must be an empty line here."
    **Transition:**
    [Conversational bridge sentence + next theme’s first question]
    """
    
    session.analysis_transition = analyze_chunk(
        transition_prompt, recent_text_transition, session.custom_prompt
    )
    
    # 3. Empathy
    recent_text_empathy = " ".join(session.transcript_history[-WINDOW_SIZE_EMPATHY:])
    empathy_prompt = """
    You are an emotional support classifier.
    Determine if there is any distress from the interviewee.
    If None, 'Status: Normal'. 
    Else brief empathetic response.
    Keep it short, our goal here is to achieve empathy neutrality and calm interviewee down while we continue.
    **Note:** Adjust your response style based on any 'New Instructions' provided.
    """

    session.analysis_empathy = analyze_chunk(
        empathy_prompt, recent_text_empathy, session.custom_prompt)

# ---------------------------------------------------------
# SIMULATION & LIVE LOOPS
# ---------------------------------------------------------
def simulation_loop(file_path):
    print(f"🧪 Starting Simulation: {file_path}")
    if not os.path.exists(file_path): return

    try:
        from pydub import AudioSegment
        audio = AudioSegment.from_file(file_path)
        chunk_len = BATCH_SECS * 1000
        
        for i in range(0, len(audio), chunk_len):
            if session.stop_flag.is_set(): break
            
            chunk = audio[i : i + chunk_len]
            buf = io.BytesIO()
            buf.name = "chunk.wav"
            chunk.export(buf, format="wav")
            buf.seek(0)

            text = transcribe_buffer(buf)
            if text.strip():
                session.transcript_history.append(text)
                run_ai_analysis()
            
            if SIMULATION_DELAY > 0: time.sleep(SIMULATION_DELAY)
    except Exception as e:
        print(f"Sim Error: {e}")
    session.is_recording = False

def live_loop():
    def callback(indata, frames, time, status):
        if session.stop_flag.is_set(): raise sd.CallbackStop()
        # Simple numpy->wav conversion logic here (omitted for brevity, same as previous)
        # Call run_ai_analysis()
        pass 
        # (Refer to previous complete code for the full live_loop implementation if needed)

# ---------------------------------------------------------
# ENDPOINTS
# ---------------------------------------------------------
class ConfigUpdate(BaseModel):
    custom_prompt: str

@app.post("/upload_pdf")
async def upload_pdf(file: UploadFile = File(...)):
    """Reads PDF, converts to JSON Schema via LLM, saves to session."""
    try:
        # 1. Extract Text
        pdf_reader = PdfReader(file.file)
        text_content = ""
        for page in pdf_reader.pages:
            text_content += page.extract_text()
            
        # 2. Convert to JSON Schema
        system_prompt = """
        You are a Data Architect. Convert the interview plan text into this JSON structure:
        {
          "interview_guides_collection": [
            {
              "guide_name": "string",
              "themes": [
                {
                  "id": "theme_1", "title": "string", "objective": "string",
                  "questions": [
                    { "id": "t1_q1", "type": "main", "text": "string" }
                  ]
                }
              ]
            }
          ]
        }
        """
        
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Here is the interview plan:\n{text_content}"}
            ],
            response_format={ "type": "json_object" }
        )
        
        generated_json = json.loads(response.choices[0].message.content)
        session.interview_plan = generated_json # Update Global State
        
        return {"message": "Plan converted successfully", "data": generated_json}
        
    except Exception as e:
        print(e)
        raise HTTPException(status_code=500, detail="Failed to process PDF")

@app.get("/get_active_plan")
def get_active_plan():
    """Returns the current JSON plan for visualization"""
    return session.interview_plan

@app.post("/start")
def start_recording():
    if session.is_recording: return {"message": "Running"}
    session.stop_flag.clear()
    session.transcript_history = []
    
    target = simulation_loop if USE_SIMULATION else live_loop
    args = (SIMULATION_FILE_PATH,) if USE_SIMULATION else ()
    
    session.thread = threading.Thread(target=target, args=args, daemon=True)
    session.thread.start()
    session.is_recording = True
    return {"message": "Started"}

@app.post("/stop")
def stop_recording():
    session.stop_flag.set()
    if session.thread: session.thread.join()
    session.is_recording = False
    return {"message": "Stopped"}

@app.get("/status")
def get_status():
    return {
        "is_recording": session.is_recording,
        "transcript_list": session.transcript_history[-WINDOW_SIZE_FOLLOW_UP:],
        "followup": session.analysis_followup,
        "transition": session.analysis_transition,
        "empathy": session.analysis_empathy
    }

# Add this to ia_backend_simulation.py

@app.post("/update_config")
async def update_config(config: ConfigUpdate):
    """Updates the user-adjustable custom prompt string."""
    session.custom_prompt = config.custom_prompt
    print(f"✅ Config Updated: {session.custom_prompt}")
    return {"message": "Configuration updated successfully"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)