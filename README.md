# 🏥 MedEar
### Robust Medical Speech-to-Text for Telephony

[![Model](https://img.shields.io/badge/🤗%20Model-medear--whisper--medical-teal)](https://huggingface.co/ssevyana/medear-whisper-medical)
[![GitHub](https://img.shields.io/badge/GitHub-Medear-black)](https://github.com/shashanksnaik07/Medear)
[![Python](https://img.shields.io/badge/Python-3.10-blue)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104-green)](https://fastapi.tiangolo.com)

---

## The Problem

Healthcare call centers rely on speech recognition to transcribe patient conversations — but standard models fail badly on medical terminology:

```
Patient says:     "I take lisinopril and atorvastatin"

Base Whisper:     "I take Lycinoprel and Addervastatin"   ❌
MedEar:           "I take lisinopril and atorvastatin"    ✅
```

These are not cosmetic errors. A care team acting on "Addervastatin" has no idea what medication this patient is on. MedEar was built to solve this.

---

## How It Works

MedEar uses a three-layer pipeline:

```
┌─────────────────────────────────────────────────────┐
│                    Input Audio                       │
│           (MP3, WAV, M4A — any format)               │
└──────────────────────┬──────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────┐
│           8kHz G.711 μ-law Degradation               │
│     Simulates real telephony conditions              │
└──────────────────────┬──────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────┐
│          Fine-tuned Whisper Small                    │
│   Trained on 483 medical sentences · 500+ drugs      │
│       Final loss: 0.0027 · 8 epochs · A100           │
└──────────────────────┬──────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────┐
│         Phonetic Correction Engine                   │
│    321 corrections mapped from real Whisper failures │
└──────────────────────┬──────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────┐
│              Medical NER                             │
│   Drugs · Symptoms · Dosages · Allergies · Vitals    │
└──────────────────────┬──────────────────────────────┘
                       │
                       ▼
              Structured JSON Output
```

---

## Results

| Metric | Score |
|--------|-------|
| Average WER improvement | **+25%** |
| Best case WER improvement | **+50%** |
| Entity accuracy (8 accents, 8kHz telephony) | **88.8%** (71/80) |
| Drug names covered | **500+** |
| Phonetic corrections | **321** |
| Training sentences | **483** |
| Final training loss | **0.0027** |
| Telephony standard | **8kHz G.711 μ-law** |

---

## WER Benchmark

| Test Case | Base WER | MedEar WER | Improvement |
|-----------|----------|------------|-------------|
| Diabetes + hypertension (clean) | 23.5% | 11.8% | **+50%** |
| Diabetes + hypertension (8kHz) | 23.5% | 11.8% | **+50%** |
| Cardiac symptoms (clean) | 25.0% | 25.0% | 0%* |
| Cardiac symptoms (8kHz) | 25.0% | 25.0% | 0%* |
| Infection (clean) | 21.1% | 15.8% | **+25%** |
| Infection (8kHz) | 21.1% | 15.8% | **+25%** |
| **Average** | **23.2%** | **17.5%** | **+25%** |

*Drug extraction correct in cardiac case — WER difference is "milligrams" vs "mg" spelling.

---

## Accent Coverage

All tests on 8kHz G.711 μ-law telephony audio.

| Accent | Male | Female |
|--------|------|--------|
| US | 100% | 100% |
| British | 80% | 90% |
| Indian | 90% | 90% |
| Australian | 80% | 80% |
| **Overall** | | **88.8%** |

---

## Extracted Entities

| Entity | Examples |
|--------|---------|
| 💊 Drugs | metformin, lisinopril, atorvastatin, tirzepatide, omeprazole |
| 🤒 Symptoms | chest pain, nausea, dizziness, shortness of breath, fatigue |
| 💉 Dosages | 500 milligrams, 10 milligrams, 40 mg |
| 🕐 Frequency | twice daily, once daily, at bedtime, every morning |
| ⚠️ Allergies | peanut allergy, pine tree allergy, sulfa allergy |
| 📊 Vitals | weight (lbs), height, weight lost, goal weight |

---

## Phonetic Corrections

We systematically tested Whisper on all 500+ drug names across 8 accents. For every failure we recorded the exact mispronunciation and built a correction map — 321 corrections total.

| Whisper Output | Correct Drug |
|---------------|--------------|
| met for men | metformin |
| Lycinoprel / lysineopril | lisinopril |
| a Torvastatin / Addervastatin | atorvastatin |
| a Moxicillin | amoxicillin |
| levothoroxin | levothyroxine |
| erasipotide / oreazipotide | tirzepatide |
| listen or fill | lisinopril |
| warfare in | warfarin |
| methamorphin | metformin |
| on dancitron | ondansetron |

---

## Setup

```bash
# 1. Clone
git clone https://github.com/jem-thanmay/MedEar.git
cd Medear

# 2. Install dependencies
pip install fastapi uvicorn python-multipart twilio
pip install openai-whisper static-ffmpeg soundfile numpy==1.26.4
pip install transformers==4.36.2 tokenizers==0.15.2 accelerate==0.26.1
pip install huggingface_hub jiwer

# 3. Download fine-tuned model
python -c "
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id='ssevyana/medear-whisper-medical',
    local_dir='carecaller/models/whisper-medical'
)
print('Model downloaded')
"

# 4. Start API server
uvicorn carecaller.api.main:app --port 8000

# 5. Open dashboard
open index.html
```

---

## API Reference

### GET /health
```bash
curl http://localhost:8000/health
# {"status": "healthy"}
```

### POST /transcribe

```bash
curl -X POST http://localhost:8000/transcribe \
  -F "file=@patient_call.mp3"
```

**Response:**
```json
{
  "success": true,
  "data": {
    "base_transcript": "patient takes Lycinoprel 10mg once daily...",
    "medear_transcript": "patient takes lisinopril 10 milligrams once daily...",
    "corrected_transcript": "patient takes lisinopril 10 milligrams once daily...",
    "entities": {
      "drugs": ["lisinopril", "metformin"],
      "symptoms": ["chest pain", "nausea"],
      "dosages": ["10 milligrams", "500 milligrams"],
      "frequency": ["twice daily", "once daily"],
      "allergies": ["peanut allergy"],
      "vitals": {
        "weight_lbs": "200 lbs",
        "height": "5'9\""
      }
    }
  }
}
```

**Supported formats:** MP3, WAV, M4A, OGG, FLAC

---

## Project Structure

```
Medear/
├── carecaller/
│   ├── api/
│   │   ├── main.py          # FastAPI server
│   │   └── pipeline.py      # MedEar pipeline
│   └── models/
│       └── whisper-medical/ # Fine-tuned model (download from HuggingFace)
├── index.html               # Dashboard UI
├── Dockerfile               # Deployment
├── requirements.txt
└── README.md
```

---

## Training Details

| Parameter | Value |
|-----------|-------|
| Base model | openai/whisper-small (241M params) |
| Hardware | NVIDIA A100 80GB PCIe, CUDA 12.1 |
| Training time | ~25 minutes |
| Epochs | 8 |
| Batch size | 8 |
| Learning rate | 1e-5 |
| Optimizer | AdamW |
| Initial loss | 1.5772 |
| Final loss | **0.0027** |

**Loss curve:** 1.5772 → 0.8405 → 0.2228 → 0.0121 → 0.0067 → 0.0047 → 0.0034 → **0.0027**

---

## Tech Stack

| Component | Technology |
|-----------|-----------|
| STT Model | OpenAI Whisper Small |
| ML Framework | PyTorch 2.1 + HuggingFace Transformers 4.36.2 |
| Training Hardware | NVIDIA A100 80GB PCIe |
| Audio Processing | FFmpeg + audioop (G.711 μ-law) |
| TTS for training data | edge-tts (Microsoft Neural, 8 voices) |
| API | FastAPI + uvicorn |
| Frontend | Vanilla HTML/JS |
| Metrics | jiwer (WER + CER) |

---

## Author

**Thanmay Jembige**