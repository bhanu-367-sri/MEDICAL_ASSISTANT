# MedVision AI — Medical Diagnosis Chatbot Hackathon Starter

A Streamlit prototype that combines specialist YOLO detection, Gemini-generated educational explanations, multilingual text responses, audio input, and text-to-speech.

> **Safety:** This is a hackathon/educational prototype, not a medical device. It does not establish diagnoses, and no trained medical weights are included. Never use its output to make treatment decisions.

## 1. Requirements

- Python 3.10 or 3.11 recommended
- Internet access for installing packages and Gemini/gTTS requests
- A Gemini API key for LLM features
- Optional: your own trained YOLO weights for the specialist detection features

## 2. Setup on Windows (PowerShell)

Open this project folder in VS Code, then run:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
Copy-Item .env.example .env
```

Edit `.env` and replace the placeholder with your Gemini API key:

```text
GEMINI_API_KEY=your_real_key_here
GEMINI_MODEL=gemini-2.5-flash
```

Keep `.env private. Never upload it to GitHub or include it in screenshots.

## 3. Start the app

```powershell
streamlit run app.py
```

Streamlit will print a local URL, usually `http://localhost:8501`.

## 4. Enable specialist YOLO detection

Create a `models` folder (already included) and place your trained weights there:

| File | Intended input |
|---|---|
| `models/eye_yolo.pt` | Eye images |
| `models/lung_yolo.pt` | Chest X-rays |
| `models/fracture_yolo.pt` | Bone X-rays |

Weights must match the model labels and detection task expected by the app. The app will still start without weights, but it will clearly report that detection is unavailable. Gemini image explanation is not a substitute for a trained detector or clinician review.

### Training a model

1. Obtain a dataset you are legally allowed to use and ensure its labels are reviewed.
2. Split by patient (not merely by image) into train/validation/test to reduce leakage.
3. Prepare the dataset in Ultralytics YOLO format with a `data.yaml` file.
4. Train one specialist at a time, for example:

```powershell
python train_yolo.py --data datasets/fracture/data.yaml --name fracture_demo --epochs 50
```

5. Review precision, recall, mAP50, mAP50-95, class-level errors, and false negatives on an untouched test set.
6. Copy the selected `best.pt` into `models/fracture_yolo.pt` (and repeat for each specialty).

Do not advertise a performance score until you have measured it on your own held-out test set. Medical datasets may have licensing, privacy, and research-use restrictions.

## 5. Features

- Streamlit image upload and YOLO output visualization
- Three configurable specialist model slots
- Gemini explanations in selected languages
- Text health Q&A
- Recorded audio question input (browser/device support required)
- Text-to-speech using gTTS
- Confidence threshold control
- Safety messaging and no-model fallback

## 6. Current limitations and suggested hackathon improvements

- **Automatic modality routing:** current prototype asks the user to select the image type. To truly automate this, train a separate image-modality classifier and evaluate its confusion matrix across eye photos, X-rays, CT, and MRI.
- **CT/MRI:** these can be multi-slice studies, not ordinary single images. A robust implementation needs DICOM handling, metadata safeguards, and validated 2D/3D methods.
- **Symptom checker:** Gemini answers general health questions, but this is not a validated triage system.
- **Security:** add file-size limits, authentication, rate limiting, audit logging without sensitive data, and secure hosting before any public deployment.
- **Privacy:** do not upload identifiable medical images or patient information to third-party APIs without proper authorization, consent, and compliance review.
- **Clinical validation:** involve qualified clinicians and document intended use, limitations, dataset provenance, subgroup performance, and failure cases.

## 7. Suggested live demo

1. Launch the app and explain the safety banner.
2. Ask a general health-information question in English or Telugu.
3. Upload a sample image and choose its modality.
4. If a validated demo weight is installed, show the YOLO boxes, labels, and confidence.
5. Ask Gemini to explain what the model output means and what it cannot establish.
6. Demonstrate voice input and text-to-speech.
7. Present measured test metrics and failure cases from your dataset.

## Troubleshooting

- `GEMINI_API_KEY` missing: copy `.env.example` to `.env`, add your key, restart Streamlit.
- `No trained model found`: install your specialist `.pt` file under `models/` with the exact filename above.
- `ModuleNotFoundError`: activate `.venv` and run `pip install -r requirements.txt`.
- PowerShell blocks activation: run `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass`, then activate again, if permitted by your device policy.
- Gemini model unavailable: update `GEMINI_MODEL` to a model enabled for your API key.
