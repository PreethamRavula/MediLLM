import os
import sys
import time
import gradio as gr
import pandas as pd
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
prediction_log = []


def classify(mode, emr_text, image):
    if mode not in model_cache:
        model_cache[mode] = load_model(mode, MODEL_PATHS[mode])
    model = model_cache[mode]
    pred_text, cam_image, token_attn = predict(model, mode, emr_text=emr_text, image=image)

    # Save image to file if uploaded
    img_rel_path = None
    img_abs_path = None
    if image is not None:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        img_rel_path = f"app/demo/uploads/xray_{timestamp}.png"
        img_abs_path = os.path.abspath(img_rel_path)
        os.makedirs(os.path.dirname(img_abs_path), exist_ok=True)
        image.Save(img_abs_path)

    # Append to log
    prediction_log.append({
        "mode": mode,
        "emr": emr_text,
        "image_path": img_rel_path,  # logged as relative path
        "prediction": pred_text
    })

    return pred_text, cam_image, token_attn


def export_csv(filename):
    if not filename.strip():
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"demo_{timestamp}.csv"
    elif not filename.endswith(".csv"):
        filename += ".csv"

    csv_path = os.path.abspath(os.path.join("app/demo/exports", filename))
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)

    df = pd.DataFrame(prediction_log)
    df.to_csv(csv_path, index=False)

    return csv_path


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

    # CSV Export UI
    gr.Markdown("### 📁 Export Prediction Log")

    with gr.Row():
        filename_input = gr.Textbox(label="CSV filename (optional)", placeholder="e.g., my_predictions.csv")
        download_btn = gr.Button("Export CSV")
        csv_output = gr.File(label="Download Link")

    download_btn.click(
        fn=export_csv,
        inputs=[filename_input],
        outputs=[csv_output]
    )

if __name__ == "__main__":
    for mode, path in MODEL_PATHS.items():
        if not os.path.exists(path):
            print(f"❌ Missing model for mode {mode}: {path}")
            print("Please download or train your models before launching the demo.")
            exit(1)
    demo.launch()
