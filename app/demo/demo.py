import sys
import gradio as gr
from pathlib import Path

# Adds root directory to sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(ROOT_DIR))

from app.utils.inference_utils import load_model, predict

# Initial default values
DEFAULT_MODE = "multimodal"
MODEL_PATHS = {
    "text": "medi_llm_state_dict_text.pth",
    "image": "medi_llm_state_dict_image.pth",
    "multimodal": "medi_llm_state_dict_multimodal.pth"
}

model_cache = {}


def classify(mode, emr_text, image):
    if mode not in model_cache:
        model_cache[mode] = load_model(mode, MODEL_PATHS[mode])
    model = model_cache[mode]
    return predict(model, mode, emr_text=emr_text, image=image)


demo = gr.Interface(
    fn=classify,
    inputs=[
        gr.Radio(choices=["text", "image", "multimodal"], value=DEFAULT_MODE, label="Select Mode"),
        gr.Textbox(lines=6, label="EMR Text"),
        gr.Image(type="pil", label="Chest X-ray")
    ],
    outputs=gr.Text(label="Predicted Triage Level"),
    title="🩺 Medi-LLM: Multimodal Clinical Triage Assistant 🩻",
    description="Select a mode and input either EMR text, X-ray image, or both to get triage predictions."
)

if __name__ == "__main__":
    demo.launch()
