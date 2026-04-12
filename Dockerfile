FROM python:3.10-slim

RUN useradd -m -u 1000 user
USER user
ENV PATH="/home/user/.local/bin:$PATH"

WORKDIR /app

COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY --chown=user . /app

RUN python -c "import whisper; whisper.load_model('small'); print('Base Whisper downloaded')"

RUN python -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='ssevyana/medear-whisper-medical', local_dir='carecaller/models/whisper-medical'); print('MedEar model downloaded')"

EXPOSE 7860

CMD ["uvicorn", "carecaller.api.main:app", "--host", "0.0.0.0", "--port", "7860"]
