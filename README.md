---
title: MedEar
emoji: 🏥
colorFrom: teal
colorTo: green
sdk: docker
pinned: false
---

# MedEar 🏥
### Robust Medical Speech-to-Text for Telephony
**CareCaller AI Track | VillageHacks 2026**

## Download the Model
Model hosted on Hugging Face:
https://huggingface.co/ssevyana/medear-whisper-medical

## Setup
```bash
git clone https://github.com/shashanksnaik07/Medear.git
cd Medear
uvicorn carecaller.api.main:app --port 8000
open index.html
```

## API
POST /transcribe — upload audio, get structured medical JSON

Built at VillageHacks 2026 | CareCaller AI Track
