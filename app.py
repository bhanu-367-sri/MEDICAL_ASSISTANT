import os
import base64
from pathlib import Path
from io import BytesIO

import streamlit as st
from PIL import Image
from dotenv import load_dotenv
from ultralytics import YOLO
from google import genai
from gtts import gTTS

# Load .env from this file's directory, regardless of the terminal's current folder.
APP_DIR = Path(__file__).resolve().parent
ENV_FILE = APP_DIR / ".env"
MODEL_DIR = APP_DIR / "models"
load_dotenv(dotenv_path=ENV_FILE, override=True)

def get_model_name():
    try:
        if "GEMINI_MODEL" in st.secrets:
            return str(st.secrets["GEMINI_MODEL"]).strip()
    except Exception:
        pass
    return os.getenv("GEMINI_MODEL", "gemini-3.5-flash").strip()


MODEL_NAME = get_model_name()

st.set_page_config(page_title="MedVision AI", page_icon="🩺", layout="wide")

SPECIALISTS = {
    "Eye image": {"file": "eye_yolo.pt", "description": "Eye-image specialist"},
    "Chest X-ray": {"file": "lung_yolo.pt", "description": "Chest X-ray specialist"},
    "Bone X-ray": {"file": "fracture_yolo.pt", "description": "Bone fracture specialist"},
}
LANGUAGES = {
    "English": "en", "Hindi": "hi", "Telugu": "te", "Tamil": "ta",
    "Kannada": "kn", "Malayalam": "ml", "Bengali": "bn", "Marathi": "mr",
}


def get_api_key():
    """Read the Gemini API key from Streamlit secrets or .env file."""
    try:
        if "GEMINI_API_KEY" in st.secrets:
            return str(st.secrets["GEMINI_API_KEY"]).strip()
    except Exception:
        pass
    load_dotenv(dotenv_path=ENV_FILE, override=True)
    return os.getenv("GEMINI_API_KEY", "").strip()



def get_specialist_model_path(specialist_name):
    """Find the specialist YOLO model file, checking models/ and alternate names."""
    config = SPECIALISTS.get(specialist_name)
    if not config:
        return None
    file_name = config["file"]
    # Check standard path in models/
    standard_path = MODEL_DIR / file_name
    if standard_path.exists() and standard_path.stat().st_size > 0:
        return standard_path
    # Check alternate name typo (e.g. fracture_rolo.pt)
    alt_name = file_name.replace("_yolo.pt", "_rolo.pt")
    alt_path = MODEL_DIR / alt_name
    if alt_path.exists() and alt_path.stat().st_size > 0:
        return alt_path
    # Check workspace root
    root_path = APP_DIR / file_name
    if root_path.exists() and root_path.stat().st_size > 0:
        return root_path
    return standard_path


@st.cache_resource
def get_gemini_client(api_key):
    # Pass the key explicitly; don't rely on SDK environment auto-detection.
    client = genai.Client(api_key=api_key)
    # Disable infinite retry on 429 rate limits so UI doesn't freeze
    if hasattr(client.interactions, "sdk_configuration"):
        client.interactions.sdk_configuration.retry_config = None
    return client


def require_gemini_client():
    api_key = get_api_key()
    if not api_key:
        raise ValueError(
            f"GEMINI_API_KEY was not found. Add it to this file: {ENV_FILE}"
        )
    return get_gemini_client(api_key)


@st.cache_resource
def load_yolo_model(path_string):
    return YOLO(path_string)



def safe_yolo_predict(image, model_path, confidence):
    model = load_yolo_model(str(model_path))
    result = model.predict(source=image, conf=confidence, verbose=False)[0]
    detections = []
    if result.boxes is not None:
        for box in result.boxes:
            class_id = int(box.cls[0].item())
            detections.append({
                "label": result.names.get(class_id, str(class_id)),
                "confidence": round(float(box.conf[0].item()), 4),
                "box": [round(float(v), 1) for v in box.xyxy[0].tolist()],
            })
    annotated = Image.fromarray(result.plot()[:, :, ::-1])
    return detections, annotated


def image_to_interactions_content(image):
    """Convert a PIL image to the documented Interactions API image input shape."""
    buffer = BytesIO()
    image.convert("RGB").save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return {"type": "image", "data": encoded, "mime_type": "image/png"}


def ask_gemini(prompt, image=None, language="English", audio_bytes=None,
               audio_mime="audio/wav", selected_model=None):
    safety_rule = (
        "You are an educational health-information assistant, not a doctor. "
        "Do not claim a confirmed diagnosis or prescribe medication or dosages. "
        "Explain that AI image results can be wrong and require review by a qualified clinician. "
        "Give general next steps and urgent warning signs when relevant. "
        "For severe breathing difficulty, stroke symptoms, major trauma, sudden vision loss, "
        "or severe chest pain, advise immediate emergency care. "
        f"Respond in {language}, using plain language and clearly stating uncertainty."
    )

    contents = [
        {"type": "text", "text": safety_rule + "\n\nUser request:\n" + prompt}
    ]

    if image is not None:
        contents.append(image_to_interactions_content(image))

    if audio_bytes is not None:
        contents.append({
            "type": "audio",
            "data": base64.b64encode(audio_bytes).decode("utf-8"),
            "mime_type": audio_mime or "audio/wav",
        })
        contents.append({
            "type": "text",
            "text": "Transcribe the audio and answer the health question it contains. "
                    "If unclear, ask the user to type it."
        })

    primary_model = selected_model or MODEL_NAME
    models_to_try = [primary_model]
    for fallback in ["gemini-3.5-flash", "gemini-3.1-flash-lite", "gemini-3.8-flash"]:
        if fallback not in models_to_try:
            models_to_try.append(fallback)

    try:
        client = require_gemini_client()
    except Exception as exc:
        return str(exc)

    last_error = ""
    for candidate in models_to_try:
        try:
            response = client.interactions.create(
                model=candidate,
                input=contents,
            )
            out = response.output_text
            if out:
                if candidate != primary_model:
                    return f"*(Notice: Used fallback model `{candidate}` because `{primary_model}` hit rate limit/unavailable)*\n\n{out}"
                return out
            return "Gemini returned no text. Please try again."
        except Exception as exc:
            last_error = str(exc)
            if any(k in last_error for k in ["429", "Rate limit", "too_many_requests", "NOT_FOUND", "not available"]):
                continue
            break

    if "429" in last_error or "Rate limit" in last_error or "too_many_requests" in last_error:
        return (
            f"Rate limit exceeded on Free Tier for Gemini models. "
            "Please wait for your daily quota to reset or supply a new key at https://aistudio.google.com/apikey.\n\n"
            f"Error details: {last_error}"
        )
    if "API_KEY_INVALID" in last_error or "API key not valid" in last_error:
        return (
            "Google rejected this API key. Create/check a key at "
            "https://aistudio.google.com/apikey, update .env, and restart the app."
        )
    if "No API key was provided" in last_error or "GEMINI_API_KEY was not found" in last_error:
        return f"Gemini API key is missing. Check this file: {ENV_FILE}"
    if "NOT_FOUND" in last_error or "not available" in last_error:
        return (
            f"Model '{primary_model}' is not available for this API key/account. "
            "Check available models in Google AI Studio and update GEMINI_MODEL in .env."
        )
    return f"Gemini request failed: {last_error}"



def speech_bytes(text, language):
    try:
        audio = BytesIO()
        gTTS(text=text[:4500], lang=LANGUAGES.get(language, "en")).write_to_fp(audio)
        audio.seek(0)
        return audio.read()
    except Exception:
        return None


st.title("🩺 MedVision AI")
st.caption("Hackathon prototype • Medical-image AI + conversational health information")
st.warning(
    "Research/demo use only — not for diagnosis or treatment. Do not upload identifying patient "
    "information. A qualified healthcare professional must review all medical images and results."
)

with st.sidebar:
    st.header("Settings")
    language = st.selectbox("Response language", list(LANGUAGES.keys()), index=0)
    confidence = st.slider("YOLO confidence threshold", 0.10, 0.90, 0.25, 0.05)
    model_choices = ["gemini-3.5-flash", "gemini-3.8-flash", "gemini-3.1-flash-lite"]
    model_idx = model_choices.index(MODEL_NAME) if MODEL_NAME in model_choices else 0
    selected_model = st.selectbox("Gemini model", model_choices, index=model_idx, help="Choose the active Gemini model.")
    st.markdown("---")
    st.write("**Configured specialist models**")
    for title, config in SPECIALISTS.items():
        m_path = get_specialist_model_path(title)
        has_model = m_path and m_path.exists() and m_path.stat().st_size > 0
        st.write(("✅ " if has_model else "⚪ ") + title)
    st.caption(f"Active model: {selected_model}")
    if get_api_key():
        st.success("GEMINI_API_KEY found in .env")
    else:
        st.error(f"GEMINI_API_KEY missing. Check {ENV_FILE}")

tab_image, tab_chat, tab_voice, tab_about = st.tabs(
    ["🖼️ Image analysis", "💬 Health chat", "🎙️ Voice input", "ℹ️ Model & safety"]
)

with tab_image:
    st.subheader("Medical image analysis")
    image_type = st.selectbox(
        "Choose image type / specialist",
        ["Auto (choose manually if uncertain)", *SPECIALISTS.keys()],
        help="Auto currently asks you to choose a modality; automatic routing needs a separate validated classifier."
    )
    uploaded = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg", "webp"])
    question = st.text_area(
        "Optional context (avoid names or personal identifiers)",
        placeholder="e.g., This is a chest X-ray taken during a routine check."
    )

    if uploaded:
        try:
            pil_image = Image.open(uploaded).convert("RGB")
            st.image(pil_image, caption="Uploaded image", use_container_width=True)

            selected_specialist = image_type
            if selected_specialist.startswith("Auto"):
                selected_specialist = st.radio(
                    "Select the best matching image type",
                    list(SPECIALISTS.keys()),
                    horizontal=True,
                )

            model_path = get_specialist_model_path(selected_specialist)
            col1, col2 = st.columns(2)
            detections = None

            if model_path and model_path.exists() and model_path.stat().st_size > 0:
                if st.button("Run YOLO detection"):
                    with st.spinner("Running specialist YOLO model..."):
                        try:
                            detections, annotated = safe_yolo_predict(
                                pil_image, model_path, confidence
                            )
                            st.session_state["last_detections"] = detections
                            st.session_state["last_specialist"] = selected_specialist
                            with col1:
                                st.image(
                                    annotated, caption="YOLO output",
                                    use_container_width=True
                                )
                            with col2:
                                st.markdown("**Model detections**")
                                if detections:
                                    st.dataframe(detections, use_container_width=True)
                                else:
                                    st.write(
                                        "No objects detected above the threshold. "
                                        "This does not rule out disease."
                                    )
                        except Exception as exc:
                            st.error(f"YOLO inference failed: {exc}")
            else:
                expected_loc = MODEL_DIR / SPECIALISTS[selected_specialist]["file"]
                st.warning(
                    f"No trained model found for `{selected_specialist}` at `{expected_loc.relative_to(APP_DIR)}`. "
                    "Add your own validated weights to enable detection."
                )

            if st.button("Generate explanation and next steps", type="primary"):
                latest_detections = st.session_state.get("last_detections")
                summary = (
                    f"Image type selected: {selected_specialist}. "
                    f"YOLO results: {latest_detections if latest_detections is not None else 'No detector result available.'}. "
                    f"User context: {question or 'None supplied'}. "
                    "Explain what these outputs may and may not mean, possible general causes, "
                    "reasonable next steps, and when to seek urgent care. Do not infer a confirmed diagnosis."
                )
                with st.spinner("Preparing cautious, educational explanation..."):
                    answer = ask_gemini(summary, image=pil_image, language=language, selected_model=selected_model)
                st.markdown("### AI-assisted explanation")
                st.write(answer)
                st.caption("LLM output may be inaccurate. Seek professional interpretation.")
                audio = speech_bytes(answer, language)
                if audio:
                    st.audio(audio, format="audio/mp3")
        except Exception as exc:
            st.error(f"Could not read this image: {exc}")

with tab_chat:
    st.subheader("Ask a general health question")
    st.caption("General information only; not a substitute for medical care.")
    if "messages" not in st.session_state:
        st.session_state.messages = []
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    user_prompt = st.chat_input("Describe a symptom or ask a general health question...")
    if user_prompt:
        st.session_state.messages.append({"role": "user", "content": user_prompt})
        with st.chat_message("user"):
            st.markdown(user_prompt)
        with st.chat_message("assistant"):
            history = "\n".join(
                f'{m["role"]}: {m["content"]}'
                for m in st.session_state.messages[-8:]
            )
            with st.spinner("Preparing response..."):
                answer = ask_gemini(history, language=language, selected_model=selected_model)
            st.markdown(answer)
            audio = speech_bytes(answer, language)
            if audio:
                st.audio(audio, format="audio/mp3")
        st.session_state.messages.append({"role": "assistant", "content": answer})

with tab_voice:
    st.subheader("Speak your question")
    st.write("Record a short question. Gemini will attempt to transcribe it and provide general information.")
    audio_input = st.audio_input("Record audio")
    if audio_input and st.button("Transcribe and answer"):
        with st.spinner("Processing audio..."):
            answer = ask_gemini(
                "Transcribe the spoken question and answer it with safe general health information.",
                language=language,
                audio_bytes=audio_input.getvalue(),
                audio_mime=audio_input.type or "audio/wav",
                selected_model=selected_model,
            )
        st.markdown("### Response")
        st.write(answer)
        audio = speech_bytes(answer, language)
        if audio:
            st.audio(audio, format="audio/mp3")

with tab_about:
    st.subheader("Prototype architecture")
    st.markdown("""
    1. **Streamlit UI** accepts image, text, and recorded audio.
    2. **Specialist router** selects an eye, chest X-ray, or bone X-ray YOLO model.
    3. **YOLO inference** returns class labels, confidence scores, and bounding boxes.
    4. **Gemini Interactions API** explains outputs in the selected language.
    5. **gTTS** reads the response aloud when the selected language is supported.

    **Important limitations**
    - No trained medical weights or dataset are bundled.
    - Train and validate each specialist model on appropriately labelled data.
    - Automatic modality routing is not implemented; the user selects the image type.
    - Report only measured performance from a held-out test set.
    - Do not use this prototype for clinical decisions or emergency triage.
    """)
