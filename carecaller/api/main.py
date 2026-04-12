
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from twilio.twiml.voice_response import VoiceResponse, Gather
import tempfile, os, shutil
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from pipeline import run_pipeline

app = FastAPI(title="MedEar API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def root():
    return {"status": "MedEar API running", "version": "1.0.0"}

@app.get("/health")
def health():
    return {"status": "healthy"}

@app.post("/transcribe")
async def transcribe(file: UploadFile = File(...)):
    suffix = os.path.splitext(file.filename)[1] or ".mp3"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name
    try:
        result = run_pipeline(tmp_path)
        return {"success": True, "data": result}
    except Exception as e:
        return {"success": False, "error": str(e)}
    finally:
        os.unlink(tmp_path)

@app.post("/twilio/incoming")
async def twilio_incoming():
    """Twilio webhook — answers the call and asks caller to speak"""
    response = VoiceResponse()
    gather = Gather(
        input="speech",
        action="/twilio/process",
        method="POST",
        speech_timeout="auto",
        language="en-US"
    )
    gather.say(
        "Welcome to MedEar. Please describe the patient medication "
        "or symptoms after the beep.",
        voice="alice"
    )
    response.append(gather)
    return str(response)

@app.post("/twilio/process")
async def twilio_process(SpeechResult: str = ""):
    """Twilio sends speech result — we run NER and read back"""
    response = VoiceResponse()

    if not SpeechResult:
        response.say("Sorry, I did not catch that. Please call back.", voice="alice")
        return str(response)

    # Run NER on speech result
    from pipeline import extract_entities
    entities = extract_entities(SpeechResult)

    # Build spoken response
    parts = []
    if entities["drugs"]:
        parts.append("Drugs detected: " + ", ".join(entities["drugs"]))
    if entities["symptoms"]:
        parts.append("Symptoms: " + ", ".join(entities["symptoms"]))
    if entities["dosages"]:
        parts.append("Dosages: " + ", ".join(entities["dosages"]))
    if entities["frequency"]:
        parts.append("Frequency: " + ", ".join(entities["frequency"]))

    if parts:
        spoken = ". ".join(parts) + ". Thank you for using MedEar."
    else:
        spoken = ("I heard: " + SpeechResult +
                  ". No medical entities were detected. Thank you.")

    response.say(spoken, voice="alice")
    return str(response)
