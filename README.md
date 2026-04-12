# MedEar 🏥
### Robust Medical Speech-to-Text for Telephony
**CareCaller AI Track | VillageHacks 2026**

[![Model on HuggingFace](https://img.shields.io/badge/🤗%20HuggingFace-Model-teal)](https://huggingface.co/shashanksnaik07/medear-whisper-medical)
[![Python 3.10](https://img.shields.io/badge/Python-3.10-blue)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104-green)](https://fastapi.tiangolo.com)

---

## The Problem

Standard speech recognition fails on healthcare calls:

| Base Whisper Output | Correct Term |
|---------------------|--------------|
| "Lycinoprel" | lisinopril |
| "Addervastatin" | atorvastatin |
| "a Moxicillin" | amoxicillin |
| "met for men" | metformin |
| "erasipotide" | tirzepatide |
| "levothoroxin" | levothyroxine |

These are not cosmetic errors — they are medication safety issues. A care team acting on "Addervastatin" would have no idea what drug this patient is taking.

---

## Our Solution

MedEar is a three-layer pipeline built specifically for telephony healthcare calls:

```
Input Audio (any format)
        ↓
8kHz G.711 μ-law Degradation
(simulates real telephony conditions)
        ↓
Fine-tuned Whisper Small
(trained on 483 medical sentences, 500+ drug names)
        ↓
Phonetic Correction Engine
(321 corrections mapped from real Whisper failures)
        ↓
Medical NER
(drugs, symptoms, dosages, frequency, allergies, vitals)
        ↓
Structured JSON Output
```

---

## Results

| Metric | Score |
|--------|-------|
| Average WER improvement | **+25%** |
| Best case WER improvement | **+50%** |
| Entity accuracy (8 accents, 8kHz) | **88.8%** (71/80) |
| Drug names covered | **500+** |
| Phonetic corrections | **321** |
| Training sentences | **483** |
| Final training loss | **0.0027** |
| Telephony standard | **8kHz G.711 μ-law** |

---

## Accent Coverage

All tests conducted on 8kHz G.711 μ-law telephony audio — the exact standard CareCaller uses.

| Accent | Gender | Accuracy |
|--------|--------|----------|
| US | Male | 100% |
| US | Female | 100% |
| British | Male | 80% |
| British | Female | 90% |
| Indian | Female | 90% |
| Indian | Male | 90% |
| Australian | Male | 80% |
| Australian | Female | 80% |
| **Overall** | | **88.8%** |

---

## WER Benchmark

| Test Case | Condition | Base WER | MedEar WER | Improvement |
|-----------|-----------|----------|------------|-------------|
| Diabetes + hypertension | Clean 16kHz | 23.5% | 11.8% | +50% |
| Diabetes + hypertension | 8kHz G.711 | 23.5% | 11.8% | +50% |
| Cardiac symptoms | Clean 16kHz | 25.0% | 25.0% | 0%* |
| Cardiac symptoms | 8kHz G.711 | 25.0% | 25.0% | 0%* |
| Infection | Clean 16kHz | 21.1% | 15.8% | +25% |
| Infection | 8kHz G.711 | 21.1% | 15.8% | +25% |
| **Average** | | **23.2%** | **17.5%** | **+25%** |

*Cardiac: drug extraction correct (atorvastatin detected), WER difference is "milligrams" vs "mg" spelling.

---

## Extracted Entities

| Entity | Examples |
|--------|---------|
| Drugs | metformin, lisinopril, atorvastatin, tirzepatide, omeprazole |
| Symptoms | chest pain, nausea, dizziness, shortness of breath, fatigue |
| Dosages | 500 milligrams, 10 milligrams, 40 mg |
| Frequency | twice daily, once daily, at bedtime, every morning |
| Allergies | peanut allergy, pine tree allergy, sulfa allergy |
| Vitals | weight (lbs), height, weight lost, goal weight |

---

## Download the Model

The fine-tuned model (922MB) is hosted on Hugging Face:

**[🤗 shashanksnaik07/medear-whisper-medical](https://huggingface.co/shashanksnaik07/medear-whisper-medical)**

```bash
# Download using huggingface_hub
pip install huggingface_hub
python -c "
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id='shashanksnaik07/medear-whisper-medical',
    local_dir='carecaller/models/whisper-medical'
)
print('Model downloaded')
"
```

---

## Setup

### 1. Clone the repo
```bash
git clone https://github.com/shashanksnaik07/Medear.git
cd Medear
```

### 2. Download the model
```bash
pip install huggingface_hub
python -c "
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id='shashanksnaik07/medear-whisper-medical',
    local_dir='carecaller/models/whisper-medical'
)
"
```

### 3. Install dependencies
```bash
pip install fastapi uvicorn python-multipart
pip install openai-whisper static-ffmpeg soundfile numpy==1.26.4
pip install transformers==4.36.2 tokenizers==0.15.2 accelerate==0.26.1
pip install jiwer pyngrok
```

### 4. Start the API server
```bash
uvicorn carecaller.api.main:app --port 8000
```

### 5. Open the dashboard
```bash
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

Upload any audio file, receive structured medical JSON.

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
        "height": "5'9\"",
        "weight_lost": "5 lbs"
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
│   │   └── pipeline.py      # MedEar pipeline (corrections + NER)
│   └── models/
│       └── whisper-medical/ # Fine-tuned model (download from HuggingFace)
└── index.html               # Dashboard (upload + live recording)
```

---

## Tech Stack

| Component | Technology | Detail |
|-----------|-----------|--------|
| STT Model | OpenAI Whisper Small | 241M parameters |
| Training Hardware | NVIDIA A100 80GB PCIe | CUDA 12.1 |
| ML Framework | PyTorch 2.1 + HuggingFace | Transformers 4.36.2 |
| Audio | FFmpeg + audioop | G.711 μ-law simulation |
| TTS (training data) | edge-tts | Microsoft Neural, 8 voices |
| API | FastAPI + uvicorn | REST, multipart upload |
| Frontend | Vanilla HTML/JS | Zero dependencies |
| Metrics | jiwer | WER + CER |

---

## Training Details

- **Base model**: openai/whisper-small
- **Training data**: 483 sentences (453 drug-specific + 30 clinical)
- **Hardware**: NVIDIA A100 80GB PCIe (~25 min training)
- **Epochs**: 8 | **Batch size**: 8 | **LR**: 1e-5 | **Optimizer**: AdamW

Loss curve:

| Epoch | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
|-------|---|---|---|---|---|---|---|---|
| Loss | 1.5772 | 0.8405 | 0.2228 | 0.0121 | 0.0067 | 0.0047 | 0.0034 | **0.0027** |

---

## How Phonetic Corrections Work

We systematically tested Whisper on all 500+ drug names across 8 accents. For every failure we recorded the exact mispronunciation and added it to the correction map. 321 corrections total.

Sample corrections:

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
| on dancitron | ondansetron |
| methamorphin | metformin |

---

## What Makes MedEar Different

1. **Systematic mispronunciation mapping** — corrections discovered by automated testing across 500+ drugs, not guesswork
2. **Real telephony simulation** — all benchmarks on 8kHz G.711 μ-law audio, not clean studio recordings
3. **Ensemble NER** — runs extraction on both base and fine-tuned outputs, merges results so nothing is missed
4. **Long call support** — automatically switches strategy for calls over 30 seconds to prevent truncation
5. **Real voice validated** — tested on actual human recordings in addition to synthetic TTS

---

## Dashboard Features

- Upload audio files (MP3, WAV, M4A, OGG, FLAC)
- Live microphone recording with animated waveform
- Side-by-side Base Whisper vs MedEar transcript
- Color-coded entity tags (drugs, symptoms, dosages, allergies, vitals)
- WER improvement bar chart
- Real-time API health indicator

---

*Built at VillageHacks 2026 | CareCaller AI Track*