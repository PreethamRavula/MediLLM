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


with gr.Blocks(theme=gr.themes.Glass(), css=".centered {text-align: center;}") as demo:
    # Centered title and subtitle
    gr.Markdown("<h2 class='centered'>🩺 Medi-LLM: Clinical Triage Assistant 🩻</h2>")
    gr.Markdown("<p class='centered'>Upload a chest X-ray and/or enter EMR text to get a triage level prediction.</p>")

    # Mode selection
    with gr.Row():
        mode = gr.Radio(["text", "image", "multimodal"], value=DEFAULT_MODE, label="Select Input Mode")

    # Input: EMR text and/or image
    with gr.Row():
        emr_text = gr.Textbox(lines=6, label="EMR Text", placeholder="Enter clinical notes here...")
        image = gr.Image(type="pil", label="Chest X-ray")

    with gr.Row():
        submit_btn = gr.Button("Run Inference")

    result = gr.Textbox(label="Prediction")

    submit_btn.click(fn=classify, inputs=[mode, emr_text, image], outputs=result)

if __name__ == "__main__":
    demo.launch()
