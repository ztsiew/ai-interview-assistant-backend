# frontend_gradio.py
import gradio as gr
import requests
import time
import json

API_URL = "http://localhost:8000"

# --- Logic Functions (Talk to Backend) ---
def upload_file(file):
    if not file: return "⚠️ No file selected."
    try:
        # Send file to backend
        with open(file, "rb") as f:
            files = {"file": f}
            res = requests.post(f"{API_URL}/upload_plan", files=files)
        return "✅ Plan Uploaded Successfully" if res.status_code == 200 else "❌ Upload Failed"
    except: return "❌ Backend Offline"

def update_prompt(text):
    try:
        requests.post(f"{API_URL}/update_config", json={"custom_prompt": text})
    except: pass

def toggle_recording(is_recording):
    if not is_recording:
        # Start
        try:
            requests.post(f"{API_URL}/start")
            return (
                True, # New recording state
                gr.update(value="🔴 STOP RECORDING", variant="stop"),
                gr.update(visible=False) # Hide scorecard
            )
        except: return False, gr.update(), gr.update()
    else:
        # Stop
        try:
            res = requests.post(f"{API_URL}/stop").json()
            score = res.get("scorecard", "No score generated.")
            return (
                False, # New recording state
                gr.update(value="▶ START RECORDING", variant="primary"),
                gr.update(value=score, visible=True) # Show scorecard
            )
        except: return True, gr.update(), gr.update()

def fetch_updates():
    try:
        res = requests.get(f"{API_URL}/status").json()
        return (
            res["transcript"] or "Listening...",
            res["followup"],
            res["transition"],
            res["empathy"]
        )
    except:
        return "Backend Offline...", "...", "...", "..."

# --- UI Layout ---
css = """
.panel { background: white; border-radius: 8px; padding: 15px; border: 1px solid #eee; box-shadow: 0 2px 4px rgba(0,0,0,0.05); }
.header { text-align: center; margin-bottom: 20px; color: #333; }
"""

with gr.Blocks(theme=gr.themes.Soft(), css=css, title="Interview Copilot") as demo:
    
    # Internal State to track if we are recording
    recording_state = gr.State(False)

    gr.Markdown("# 🎤 AI Interview Co-Pilot", elem_classes="header")
    
    with gr.Row():
        # Left: Controls
        with gr.Column(scale=1):
            with gr.Group(elem_classes="panel"):
                gr.Markdown("### ⚙️ Setup")
                plan_file = gr.File(label="Upload JSON Plan", file_types=[".json"], height=80)
                upload_status = gr.Markdown("Waiting for plan...")
                plan_file.upload(upload_file, inputs=plan_file, outputs=upload_status)
                
                custom_prompt = gr.Textbox(label="Custom Instructions", value="Keep it concise.")
                custom_prompt.change(update_prompt, inputs=custom_prompt)
            
            with gr.Group(elem_classes="panel"):
                gr.Markdown("### 🔴 Controls")
                btn_record = gr.Button("▶ START RECORDING", variant="primary")

        # Right: Live Data
        with gr.Column(scale=2):
            with gr.Group(elem_classes="panel"):
                gr.Markdown("### 📜 Live Transcript")
                box_transcript = gr.Textbox(show_label=False, lines=6, interactive=False)
            
            with gr.Tabs():
                with gr.TabItem("🔍 Follow-up"):
                    box_followup = gr.Markdown("Waiting...", elem_classes="panel")
                with gr.TabItem("➡️ Transition"):
                    box_transition = gr.Markdown("Waiting...", elem_classes="panel")
                with gr.TabItem("❤️ Empathy"):
                    box_empathy = gr.Markdown("Waiting...", elem_classes="panel")

    # Scorecard (Hidden)
    box_scorecard = gr.Markdown(visible=False, elem_classes="panel")

    # --- Event Wiring ---
    
    # 1. Start/Stop Button
    btn_record.click(
        toggle_recording, 
        inputs=[recording_state], 
        outputs=[recording_state, btn_record, box_scorecard]
    )

    # 2. Timer (Polls backend every 1s)
    timer = gr.Timer(1.0)
    timer.tick(
        fetch_updates, 
        outputs=[box_transcript, box_followup, box_transition, box_empathy]
    )

if __name__ == "__main__":
    demo.launch()