import os
import sys
import time
import shutil
import logging
import gradio as gr
import pandas as pd
from PIL import Image
from pathlib import Path
from threading import Lock

# Adds root directory to sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(ROOT_DIR))
PERSIST_ROOT = Path("/data") if Path("/data").exists() else ROOT_DIR

from app.utils.inference_utils import load_model, predict
from concurrent.futures import ThreadPoolExecutor

# if available, grab weight resolver to prefetch files
try:
    from app.utils.inference_utils import resolve_weights_path
except Exception:
    resolve_weights_path = None

# Initial default values
DEFAULT_MODE = "multimodal"
model_cache = {}
prediction_log_user = []
prediction_log_doctor = []
MODEL_MODES = ["text", "image", "multimodal"]
MODEL_LOCKS = {m: Lock() for m in MODEL_MODES}
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=LOG_LEVEL,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s"
)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("gradio").setLevel(logging.WARNING)
log = logging.getLogger("medi_llm")


def get_model(mode: str):
    # Lock per mode to avoid double-loads
    lock = MODEL_LOCKS.get(mode)
    if lock is None:
        # fallback
        if mode not in model_cache:
            model_cache[mode] = load_model(mode)
        return model_cache[mode]

    with lock:
        if mode not in model_cache:
            model_cache[mode] = load_model(mode)
        return model_cache[mode]


# Warm-start toggles (env-controlled)
WARM_START = os.getenv("WARM_START", "1") == "1"
WARM_START_MODES = [m.strip() for m in os.getenv("WARM_START_MODES", "text,image,multimodal").split(",") if m.strip()]
WARM_START_WORKERS = max(1, int(os.getenv("WARM_START_WORKERS", "3")))


def warm_start():
    if not WARM_START:
        log.info("[warm] disabled via WARM_START=0")
        return
    log.info("[warm] prefetching weights & models...")
    modes_env = WARM_START_MODES or MODEL_MODES
    modes = [m for m in modes_env if m in MODEL_MODES]

    # Ensure weights exist (local-first, Hub fallback) if resolver is available
    if resolve_weights_path is not None:
        for m in modes:
            try:
                weight_path = resolve_weights_path(m)
                log.info(f"[warm] weights ok for {m}: {weight_path}")
            except Exception as e:
                log.warning(f"[warm] weights skip {m}: {e}")

    # Build and cache models in parallel (keeps first inference snappy)
    def _load(mm):
        try:
            _ = get_model(mm)
            log.info(f"[warm] model ready: {mm}")
        except Exception as e:
            log.warning(f"[warm] model failed {mm}: {e}")

    with ThreadPoolExecutor(max_workers=min(len(modes), WARM_START_WORKERS)) as ex:
        for m in modes:
            ex.submit(_load, m)

    log.info("[warm] done")


def build_result_box(value: str = "", cls: str | None = None):
    kwargs = dict(
        value=value,
        label="🧪 Triage Prediction",
        interactive=False,
        elem_id="result_box",
    )
    if cls:
        kwargs["elem_classes"] = [cls]
    return gr.Textbox(**kwargs)


def classify(role, mode, normalize_mode, emr_text, image, use_rollout):
    grad_cam_path = "N/A"
    token_attn_path = "N/A"

    # Control output visibility
    show_tabs = (role == "Doctor")
    show_gradcam = (role == "Doctor" and mode in ["image", "multimodal"])
    show_attention = (role == "Doctor" and mode in ["text", "multimodal"])

    # ✅ Skip inference if required inputs are missing for selected mode
    no_text_needed_but_missing = (mode in ["text", "multimodal"]) and (not emr_text or not emr_text.strip())
    no_image_needed_but_missing = (mode in ["image", "multimodal"]) and (image is None)
    should_skip = (
        (mode == "text" and no_text_needed_but_missing) or (mode == "image" and no_image_needed_but_missing) or (mode == "multimodal" and no_text_needed_but_missing and no_image_needed_but_missing)
    )
    if should_skip:
        count = len(prediction_log_doctor) if role == "Doctor" else len(prediction_log_user)
        return (
            build_result_box("⚠️ Please enter EMR text or upload an image to run inference.", "prediction-none"),  # result_box
            gr.update(visible=False),                                                                              # gradcam_img
            gr.update(visible=False),                                                                              # token_attention
            gr.update(value="", visible=False),                                                                    # top5_html
            gr.update(visible=False),                                                                              # confidence_label
            gr.update(visible=False),                                                                              # insights_tab
            f"Predictions: {count}",                                                                               # prediction_count_box
            gr.update(value={}, visible=True)                                                                      # class_probs_json
        )

    # Image size guard + load
    if image is not None:
        image_path = Path(image)
        image_size = image_path.stat().st_size
        # Enforce 5MB limit (5 * 1024 * 1024 bytes)
        if image_size > 5 * 1024 * 1024:
            count = len(prediction_log_doctor) if role == "Doctor" else len(prediction_log_user)
            return (
                build_result_box("❌ Image exceeds 5MB size limit.", "prediction-none"),
                gr.update(visible=False),
                gr.update(visible=False),
                gr.update(value="", visible=False),
                gr.update(visible=False),
                gr.update(visible=False),  # Hide insights tab on error
                f"Predictions: {count}",
                gr.update(value={}, visible=True),
            )
        image = Image.open(image).convert("RGB")

    # Model caching
    model = get_model(mode)

    # Run prediction
    try:
        log.info("classify(): normalize_mode=%s | use_rollout=%s", normalize_mode, use_rollout)
        pred_text, cam_image, token_attn, confidence, probs, top5 = predict(
            model,
            mode,
            emr_text=emr_text,
            image=image,
            normalize_mode=normalize_mode,
            need_token_vis=show_attention,
            use_rollout=use_rollout,
        )

        top5 = top5 or []
    except Exception as e:
        log.exception("Inference failed")
        count = len(prediction_log_doctor) if role == "Doctor" else len(prediction_log_user)
        return (
            build_result_box(f"❌ {str(e)}", "prediction-none"),
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(value="", visible=False),
            gr.update(visible=False),
            gr.update(visible=False),
            f"Predictions: {count}",
            gr.update(value={}, visible=True)
        )

    pred_norm = (str(pred_text) if pred_text is not None else "").strip().lower()
    triage_class = {
        "high": "prediction-high",
        "medium": "prediction-medium",
        "low": "prediction-low",
    }.get(pred_norm, "prediction-low")

    # Class probabilities (ensure always 3)
    if isinstance(probs, (list, tuple)):
        if probs and isinstance(probs[0], (list, tuple)):
            flat_probs = probs[0]
        else:
            flat_probs = list(probs)
    else:
        flat_probs = []

    if len(flat_probs) != 3:
        class_probs = {"low": 0.0, "medium": 0.0, "high": 0.0}
    else:
        class_probs = {label: round(prob, 3) for label, prob in zip(["low", "medium", "high"], flat_probs)}

    # Save uploads (relative path in logs)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    img_rel_path = f"app/demo/uploads/xray_{timestamp}.png" if image else "N/A"

    # Save image to file if uploaded
    if image:
        img_abs_path = (PERSIST_ROOT / img_rel_path).resolve()
        os.makedirs(img_abs_path.parent, exist_ok=True)
        image.save(img_abs_path)

    # Save Grad-CAM if Doctor and mode uses image
    if cam_image and role == "Doctor" and mode in ["image", "multimodal"]:
        cam_rel_path = f"app/demo/exports/{role.lower()}/gradcam/gradcam_{pred_text}_{timestamp}.png"
        cam_abs_path = (PERSIST_ROOT / cam_rel_path).resolve()
        os.makedirs(cam_abs_path.parent, exist_ok=True)
        cam_image.save(cam_abs_path)
        grad_cam_path = cam_rel_path

    # Save token attention if Doctor and mode uses text
    if token_attn and role == "Doctor" and mode in ["text", "multimodal"]:
        attn_rel_path = f"app/demo/exports/{role.lower()}/tokenattention/token_attn_{pred_text}_{timestamp}.txt"
        attn_abs_path = (PERSIST_ROOT / attn_rel_path).resolve()
        os.makedirs(attn_abs_path.parent, exist_ok=True)
        with open(attn_abs_path, "w") as f:
            f.write(f"Normalization Mode: {normalize_mode}\n")
            f.write(f"Use Rollout: {use_rollout}\n")
            f.write("Token Attention (word | score):\n")
            f.write(str(token_attn) + "\n\n")
            f.write("Top 5 tokens (token | % contribution):\n")
            if top5:
                for tok, pct in top5:
                    f.write(f"{tok}\t{pct:.2f}%\n")
            else:
                f.write("(none)\n")
        token_attn_path = attn_rel_path

    # Append to log
    log_entry = {
        "mode": mode,
        "normalize_mode": normalize_mode,
        "use_rollout": bool(use_rollout),
        "emr_text": emr_text or "N/A",
        "image_path": img_rel_path if mode in ["image", "multimodal"] else "N/A",  # logged as relative path
        "prediction": pred_text,
        "confidence": round(confidence, 3),
        "grad_cam_path": grad_cam_path if role == "Doctor" else "N/A",
        "token_attention_path": token_attn_path if role == "Doctor" else "N/A",
        "top5_tokens": "; ".join([f"{tok}:{pct:.1f}%" for tok, pct in (top5 or [])])
    }

    if role == "Doctor":
        prediction_log_doctor.append(log_entry)
        count = len(prediction_log_doctor)
    else:
        prediction_log_user.append(log_entry)
        count = len(prediction_log_user)

    return (
        build_result_box(pred_text, triage_class),
        gr.update(value=cam_image, visible=show_gradcam),
        gr.update(value=token_attn, visible=show_attention),
        render_top5_html(top5),
        gr.update(value=f"{confidence:.2f}", visible=True),
        gr.update(visible=show_tabs),
        f"Predictions: {count}",
        gr.update(value=class_probs, visible=True)
    )


def render_inputs(mode):
    is_text = mode in ["text", "multimodal"]
    is_image = mode in ["image", "multimodal"]

    emr_text = gr.Textbox(
        visible=is_text,
        lines=6,
        label="EMR Text",
        placeholder="Enter clinical notes here...",
        elem_id="emr_textbox"
    )

    image = gr.Image(
        visible=is_image,
        type="filepath",
        label="Chest X-ray",
        image_mode="RGB",
        show_label=True,
        height=224,
        elem_id="xray_image"
    )

    max_note = gr.HTML(
        "<p style='font-size: 0.9em; color: #a9b1d6;'>Maximum file size: 5MB</p>",
        visible=is_image
    )

    return emr_text, image, max_note


def render_top5_html(top5):
    """
    top5: list[ (token:str, pct:float) ] where pct is 0..100
    Returns a gr.update with an HTML table colored by contribution (continuous gradient)
    """
    if not top5:
        return gr.update(value="", visible=False)

    def _lerp(a, b, t):  # linear interpolation
        return a + (b - a) * t

    def _rgb_to_hex(rgb):  # (r, g, b) -> "#rrggbb"
        r, g, b = (max(0, min(255, int(round(x)))) for x in rgb)
        return f"#{r:02x}{g:02x}{b:02x}"

    def _interp_color(stops, t):
        """
        stops: list[(pos, (r,g,b))], pos in [0,1], sorted.
        t in [0,1] -> interpolate between nearest stops
        """
        t = max(0.0, min(1.0, float(t)))
        for i in range(len(stops) - 1):
            p0, c0 = stops[i]
            p1, c1 = stops[i + 1]
            if t <= p1:
                # local interpolation factor
                if p1 == p0:
                    w = 0.0
                else:
                    w = (t - p0) / (p1 - p0)
                return (
                    _lerp(c0[0], c1[0], w),
                    _lerp(c0[1], c1[1], w),
                    _lerp(c0[2], c1[2], w),
                )
        return stops[-1][-1]

    def _text_color_for_bg(rgb):
        # YIQ luma for contrast; threshold ~128
        r, g, b = rgb
        yiq = (r * 299 + g * 587 + b * 114) / 1000.0
        return "#000000" if yiq >= 128 else "#ffffff"

    # --- gradient (low->high): green -> chartreuse -> orange -> red ---
    # tweak the mid stops to our taste
    color_stops = [
        (0.00, (27, 67, 50)),  # deep green
        (0.40, (128, 170, 30)),  # chartreuse-ish
        (0.70, (255, 165, 0)),  # orange
        (1.00, (208, 0, 0)),    # red
    ]

    # Normalize to [0, 1] on the 5 items so colors spread even if skewed
    vals = [pct for _, pct in top5]
    vmin, vmax = min(vals), max(vals)
    if vmax - vmin < 1e-9:
        norms = [0.5] * len(vals)  # all equal -> neutral middle color
    else:
        norms = [(v - vmin) / (vmax - vmin) for v in vals]

    # Build rows
    row_html = []
    for (tok, pct), t in zip(top5, norms):
        rgb = _interp_color(color_stops, t)
        bg = _rgb_to_hex(rgb)
        fg = _text_color_for_bg(rgb)
        row_html.append(
            f"<tr style='background:{bg}; color:{fg};'>"
            f"<td style='padding:10px 12px; border-bottom:1px solid rgba(255,255,255,0.06);'>{tok}</td>"
            f"<td style='padding:10px 12px; text-align:right; border-bottom:1px solid rgba(255,255,255,0.06);'>{pct:.1f}%</td>"
            "</tr>"
        )

    # color rows by normalized importance
    table = (
        "<div class='top5-box' style='margin-top:10px;'>"
        "<h4 style='margin:0 0 8px; color:#e5e7eb;'>Top 5 tokens (by contribution)</h4>"
        "<table class='top5-table' style='width:100%; border-collapse:collapse;"
        " background:#11131a; border:1px solid #2a2f3a; border-radius:10px; overflow:hidden;'>"
        "<thead>"
        "<tr style='background:#0f1320; color:#cbd5e1;'>"
        "<th style='text-align:left; padding:10px 12px; font-weight:600;'>Token</th>"
        "<th style='text-align:right; padding:10px 12px; font-weight:600;'>Contribution</th>"
        "</tr>"
        "</thead>"
        f"<tbody>{''.join(row_html)}</tbody>"
        "</table>"
        "</div>"
    )

    return gr.update(value=table, visible=True)


def export_csv(filename, role):
    records = prediction_log_doctor if role == "Doctor" else prediction_log_user
    if not records:
        # Hide the file component and show a warning
        return (
            gr.update(value=None, visible=False),                                              # csv_output
            gr.update(value="⚠️ No predictions to export.", visible=True, interactive=False),  # export_status_box
        )  # Prevent empty exports

    if not filename.strip():
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"{role.lower()}_predictions_{timestamp}.csv"
    elif not filename.endswith(".csv"):
        filename += ".csv"

    csv_path = (PERSIST_ROOT / f"app/demo/exports/{role.lower()}/{filename}").resolve()
    os.makedirs(csv_path.parent, exist_ok=True)

    df = pd.DataFrame(records)
    if role == "Doctor":
        columns = [
            "mode", "normalize_mode", "use_rollout", "emr_text", "image_path",
            "prediction", "confidence",
            "grad_cam_path", "token_attention_path",
            "top5_tokens"
        ]
    else:
        columns = ["mode", "emr_text", "image_path", "prediction", "confidence"]

    df = df[columns]
    df.to_csv(csv_path, index=False)

    return (
        gr.File(
            value=str(csv_path),
            label="📂 Download Link",
            elem_id="download_box",
            elem_classes=["download_box", "blink-csv"],
            visible=True,
        ),
        gr.update(value=f"✅ Exported to: {csv_path}", visible=True),  # status string -> goes into export_status_box
    )


def safe_delete_dir(path):
    try:
        if os.path.exists(path) and os.path.isdir(path):
            shutil.rmtree(path)
    except Exception as e:
        log.warning("Failed to delete %s: %s", path, e)


def clear_logs(role):

    def _resolve_log_path(path_str: str) -> Path:
        p = Path(path_str)
        return p if p.is_absolute() else (PERSIST_ROOT / p)

    # Step 1: Delete logged image files
    records = prediction_log_doctor if role == "Doctor" else prediction_log_user
    for entry in records:
        # Delete X-ray image if exists and not "N/A"
        if entry["image_path"] != "N/A":
            image_file_path = _resolve_log_path(entry["image_path"])
            if image_file_path.exists():
                try:
                    image_file_path.unlink()
                except Exception as e:
                    log.warning("Failed to delete image file: %s: %s", image_file_path, e)

        # Delete Grad-CAM
        if role == "Doctor" and entry.get("grad_cam_path") not in [None, "N/A"]:
            grad_path = _resolve_log_path(entry.get("grad_cam_path", "N/A"))
            if grad_path.exists():
                try:
                    grad_path.unlink()
                except Exception as e:
                    log.warning("Failed to delete Grad-CAM: %s: %s", grad_path, e)

        # Delete token attention
        if role == "Doctor" and entry.get("token_attention_path") not in [None, "N/A"]:
            attn_path = _resolve_log_path(entry.get("token_attention_path", "N/A"))
            if attn_path.exists():
                try:
                    attn_path.unlink()
                except Exception as e:
                    log.warning("Failed to delete token attention: %s: %s", attn_path, e)

    # Step 2: Delete folders safely
    if role == "Doctor":
        safe_delete_dir(PERSIST_ROOT / "app/demo/uploads")
        safe_delete_dir(PERSIST_ROOT / "app/demo/exports/doctor/gradcam")
        safe_delete_dir(PERSIST_ROOT / "app/demo/exports/doctor/tokenattention")
        safe_delete_dir(PERSIST_ROOT / "app/demo/exports/doctor")
    else:
        safe_delete_dir(PERSIST_ROOT / "app/demo/exports/user")
        safe_delete_dir(PERSIST_ROOT / "app/demo/uploads")

    # Step 3: Clear in-memory logs
    prediction_log_doctor.clear() if role == "Doctor" else prediction_log_user.clear()

    return gr.update(value="Predictions: 0", interactive=False)


# Confirm before clearing logs
def confirm_clear():
    return gr.update(
        value="⚠️ Are you sure you want to clear the logs? Click again to confirm.",
        visible=True,
        interactive=False,
    )


def clear_confirmed(role):
    cleared = clear_logs(role)
    return (
        cleared,
        gr.update(value="✅ Logs cleared successfully!", visible=True),
        gr.update(value=None, visible=False),  # csv_output
        gr.update(interactive=True)  # filename_input
    )


def reset_confirm_box():
    return gr.update(value="", visible=False)


def disable_filename_input():
    return gr.update(interactive=False)


def show_loading_msg():
    return gr.update(value="⏳ Running inference...", visible=True)


def update_role_state(r):
    # hide insights + token box when switching to User
    tabs_vis = (r == "Doctor")
    return (
        r,                                   # role_state
        gr.update(visible=tabs_vis),         # normalize_mode_column
        gr.update(visible=tabs_vis),         # insights_tab
        gr.update(visible=False),            # token_attention
        gr.update(visible=False),            # gradcam_img
        gr.update(visible=tabs_vis),         # use_rollout,
        gr.update(visible=False),            # top5_html
    )


def rerun_if_done(ran, role, mode, normalize_mode, emr_text, image, use_rollout):
    # If inference hasn't run yet, or we're not in Doctor mode, Do nothing.
    # Returning gr.update() for each output preserves current values & visibility
    if (not ran) or (role != "Doctor"):
        return (
            gr.update(),  # result_box
            gr.update(),  # gradcam_img
            gr.update(),  # token_attention
            gr.update(),  # top5_html
            gr.update(),  # confidence_label
            gr.update(),  # insights_tab
            gr.update(),  # prediction_count_box
            gr.update(),  # class_probs_json
        )
    # Let classify() run if already inferred once
    return classify(role, mode, normalize_mode, emr_text, image, use_rollout)


def inject_tooltips():
    return gr.HTML(
        """
        <script>
        const observer = new MutationObserver(() => {
            document.querySelectorAll(".token-attn-box .token").forEach(token => {
                const text = token.innerText;
                const pipeIndex = text.indexOf("|");
                if (pipeIndex > -1) {
                    const display = text.slice(0, pipeIndex).trim();
                    const tooltip = text.slice(pipeIndex + 1).trim();
                    token.innerText = display;
                    token.setAttribute("data-tooltip", tooltip);
                }
            });
        });
        observer.observe(document.body, { childList: true, subtree: true });
        </script>
        """
    )


def reset_ui():
    is_text = DEFAULT_MODE in ["text", "multimodal"]
    is_image = DEFAULT_MODE in ["image", "multimodal"]

    return (
        # Inputs (text/image areas)
        gr.update(value="", visible=is_text),                  # emr_text
        gr.update(value=None, visible=is_image),               # image
        gr.update(visible=is_image),                           # max_file_note

        # Prediction/result area
        build_result_box("", "prediction-none"),               # result_box - Clears glow
        gr.update(value=None, visible=False),                  # gradcam_img
        gr.update(value=None, visible=False),                  # token_attention
        gr.update(value="", visible=False),                    # top5_html
        gr.update(value="", visible=False),                    # confidence_label
        gr.update(visible=False),                              # insights_tab
        gr.update(value={}, visible=True),                     # class_probs_json
        gr.update(value="Predictions: 0", interactive=False),  # prediction_count_box

        # Role/mode controls + states
        "User",                                                # role_state
        DEFAULT_MODE,                                          # mode_state
        "visual",                                              # normalization_mode_state
        gr.update(value="User"),                               # role (radio)
        gr.update(value=DEFAULT_MODE),                         # mode (radio)
        gr.update(value="visual"),                             # normalize_mode (radio)
        gr.update(visible=False),                              # normalize_mode_column (hide in User)
        gr.update(visible=False),                              # use_rollout
        False,                                                 # rollout_state

        # Loading + inference state
        gr.update(value="", visible=False),                    # loading_msg
        False,                                                 # inference_done
        gr.update(value="", visible=False),                    # export_status_box
        gr.update(value=None, visible=False),                   # csv_output
        gr.update(interactive=True, value="")                  # filename_input
    )


def build_ui():
    # Load CSS safely (don't crash if file is missing on remote)
    style_path = Path(__file__).resolve().parent / "style.css"
    custom_css = style_path.read_text(encoding="utf-8") if style_path.exists() else ""
    log.debug("Loaded CSS bytes: %d", len(custom_css))
    with gr.Blocks(css=custom_css) as demo:
        # ----- Header -----
        gr.Markdown("## 🩺 Medi-LLM: Clinical Triage Assistant 🩻", elem_id="title")
        gr.Markdown("Upload a chest X-ray and/or enter EMR text to get a triage level prediction.", elem_id="subtitle")
        gr.HTML(
            """
            <div class='welcome-banner' style="background-color: #24283b; border-left: 4px solid #7aa2f7; padding: 16px; border-radius: 8px; margin-bottom: 16px;">
             <h3 style="margin-top: 0; color: #c0caf5;">👋 Welcome to Medi-LLM</h3>
             <p style="color: #a9b1d6; line-height: 1.6;">
               This AI assistant helps triage patients using <strong>EMR text</strong> and <strong>chest X-rays</strong>.<br>
               📝 Enter EMR notes, 📷 upload a chest X-ray, or use both for a multimodal diagnosis.<br>
               👩‍⚕️ Select <strong>Doctor</strong> mode to view insights like Grad-CAM heatmaps and token-level attention.<br>
               💾 Save your results for later by exporting them to a CSV file.
             </p>
            </div>
            """
        )

        # ----- Hidden State -----
        role_state = gr.State(value="User")
        mode_state = gr.State(value=DEFAULT_MODE)
        rollout_state = gr.State(value=False)
        normalize_mode_state = gr.State(value="visual")
        inference_done = gr.State(value=False)

        # ----- Role and Mode selection -----
        with gr.Row(equal_height=True):
            with gr.Column():
                role = gr.Radio(["User", "Doctor"], value="User", label="Select Role", info="Doctors see insights like Grad-CAM and token attention", elem_id="role_selector")
                mode = gr.Radio(["text", "image", "multimodal"], value=DEFAULT_MODE, label="Select Input Mode", info="Choose Diagnosis input type", elem_id="mode_selector")
                with gr.Column(visible=False) as normalize_mode_column:
                    normalize_mode = gr.Radio(
                        ["visual", "probabilistic"],
                        value="visual",
                        label="Attention Normalization",
                        info="Softmax sums to 1 (probabilistic). Visual uses gamma-boosted scaling for color clarity."
                    )
                    use_rollout = gr.Checkbox(
                        label="Use attention rollout (CLS -> inputs)",
                        value=False,
                        info="Includes residuals and multiplies attention across layers. Slower but often more faithful."
                    )

        # ----- Inputs -----
        with gr.Row():
            with gr.Column(scale=3, elem_id="text_col"):
                emr_text, image, max_file_note = render_inputs(DEFAULT_MODE)

        # ----- Actions -----
        with gr.Row():
            submit_btn = gr.Button(
                "🔍 Run Inference",
                elem_id="inference_btn"
            )
            reset_btn = gr.Button(
                "↩️ Reset",
                elem_id="reset_btn"
            )

        # ----- Outputs -----
        with gr.Column(elem_classes=["output-box"]):
            result_box = gr.Textbox(label="🧪 Triage Prediction", interactive=False, elem_id="result_box")
            confidence_label = gr.Label(label="📊 Confidence", visible=False)
            prediction_count_box = gr.Textbox(value="Predictions: 0", interactive=False, label="🧮 Count", elem_id="prediction_count_box")
            insights_tab = gr.Tabs(visible=False)
            class_probs_json = gr.JSON(label="🔍 Class Probabilities", visible=True, elem_classes=["json-box"])
            with insights_tab:
                with gr.Tab("📷 Grad-CAM"):
                    gradcam_img = gr.Image(visible=False, elem_classes=["gr-image-box"])
                with gr.Tab("🔬 Token Attention"):
                    token_attention = gr.HighlightedText(
                        visible=False,
                        show_legend=False,
                        color_map={
                            "0.0": "#7aa2f7",   # blue
                            "0.25": "#80deea",  # cyan
                            "0.5": "#fbc02d",   # yellow
                            "0.75": "#ff8a65",  # orange
                            "1.0": "#f7768e",   # red
                        },
                        elem_classes=["token-attn-box"]
                    )
                    top5_html = gr.HTML(value="", visible=False)

                    inject_tooltips()

                    gr.HTML("""
                    <div class="attention-legend">
                        <div style="display: flex; align-items: center; gap: 8px;">
                            <span style="font-size: 14px; color: #c0caf5;">0.0</span>
                            <div class="attention-gradient-bar"></div>
                            <span style="font-size: 14px; color: #c0caf5;">1.0</span>
                        </div>
                    </div>
                    """)

        with gr.Row():
            loading_msg = gr.Markdown(value="", visible=False, elem_classes=["loading-msg"])

        # ----- Inference Wiring -----
        submit_btn.click(
            fn=show_loading_msg,
            outputs=[loading_msg]
        ).then(
            fn=classify,
            inputs=[role_state, mode_state, normalize_mode_state, emr_text, image, rollout_state],
            outputs=[
                result_box,
                gradcam_img,
                token_attention,
                top5_html,
                confidence_label,
                insights_tab,
                prediction_count_box,
                class_probs_json,
            ]
        ).then(
            fn=lambda: gr.update(value="", visible=False),
            outputs=[loading_msg]
        ).then(
            fn=lambda: True,
            outputs=[inference_done]
        )

        # ----- Role/Mode/Param Change Wiring -----
        role.change(
            fn=update_role_state,
            inputs=[role],
            outputs=[role_state, normalize_mode_column, insights_tab, token_attention, gradcam_img, use_rollout, top5_html]
        )

        # Input Updates
        mode.change(
            fn=lambda m: (*render_inputs(m), m),
            inputs=[mode],
            outputs=[emr_text, image, max_file_note, mode_state]
        )

        normalize_mode.change(
            fn=lambda val: val,
            inputs=[normalize_mode],
            outputs=[normalize_mode_state],
            queue=False,
        )

        use_rollout.change(
            fn=lambda v: v,
            inputs=[use_rollout],
            outputs=[rollout_state],
            queue=False,
        )

        normalize_mode.change(
            fn=rerun_if_done,
            inputs=[inference_done, role_state, mode_state, normalize_mode, emr_text, image, rollout_state],
            outputs=[
                result_box,
                gradcam_img,
                token_attention,
                top5_html,
                confidence_label,
                insights_tab,
                prediction_count_box,
                class_probs_json,
            ],
            queue=False,
        )

        use_rollout.change(
            fn=rerun_if_done,
            inputs=[inference_done, role_state, mode_state, normalize_mode, emr_text, image, rollout_state],
            outputs=[
                result_box,
                gradcam_img,
                token_attention,
                top5_html,
                confidence_label,
                insights_tab,
                prediction_count_box,
                class_probs_json
            ],
            queue=False,
        )

        # ----- CSV Export & Log Controls -----
        gr.Markdown("### 📁 Export Prediction Log")

        with gr.Row(equal_height=True):
            with gr.Column(scale=3):
                filename_input = gr.Textbox(
                    label="CSV filename (optional)",
                    placeholder="e.g., triage_results.csv",
                    info="Set filename as needed or leave blank for auto-naming",
                    elem_id="csv_filename",
                )
                export_status_box = gr.Textbox(
                    value="",
                    visible=False,
                    interactive=False,
                    label="",
                    elem_id="export_status"
                )

            with gr.Column(scale=4):
                gr.Markdown(
                    "📑 **Summary**\n\nDownload your triage results for clinical review or research.",
                    elem_classes="centered"
                )
                with gr.Row():
                    with gr.Column(scale=1, min_width=200):
                        download_btn = gr.Button("💾 Export CSV", elem_id="export_button")
                    with gr.Column(scale=1, min_width=200):
                        clear_btn = gr.Button("🗑️ Clear Logs", elem_id="clear_button")
                confirm_clear_btn = gr.Button("✅ Confirm Clear", visible=False, elem_id="confirm_button")
                confirm_box = gr.Textbox(label="Status", interactive=False, visible=False, elem_id="confirm_box")

            with gr.Column(scale=3):
                csv_output = gr.File(
                    label="📂 Download Link",
                    elem_id="download_box",
                    elem_classes=["download_box"],
                    visible=False,
                    file_count="single",
                    file_types=[".csv"],
                    interactive=False,
                )

        download_btn.click(
            fn=export_csv,
            inputs=[filename_input, role_state],
            outputs=[
                csv_output,
                export_status_box
            ]
        ).then(
            fn=disable_filename_input,
            outputs=[filename_input]
        )

        clear_btn.click(
            fn=lambda: (
                confirm_clear(),
                gr.update(visible=True),
            ),
            outputs=[confirm_box, confirm_clear_btn]
        )

        confirm_clear_btn.click(
            fn=clear_confirmed,
            inputs=[role_state],
            outputs=[
                prediction_count_box,  # reset prediction count
                confirm_box,           # show success message
                csv_output,            # hide CSV output file
                filename_input         # re-enable input box
            ]
        ).then(
            fn=lambda: gr.update(visible=False),  # Hide confirm button
            outputs=[confirm_clear_btn]
        ).then(
            fn=reset_confirm_box,
            outputs=[confirm_box]
        )

        # ----- Reset Wiring -----
        reset_btn.click(
            fn=reset_ui,
            outputs=[
                emr_text,               # 1
                image,                  # 2
                max_file_note,          # 3
                result_box,             # 4
                gradcam_img,            # 5
                token_attention,        # 6
                top5_html,              # 7
                confidence_label,       # 8
                insights_tab,           # 9
                class_probs_json,       # 10
                prediction_count_box,   # 11
                role_state,             # 12
                mode_state,             # 13
                normalize_mode_state,   # 14
                role,                   # 15 (radio)
                mode,                   # 16 (radio)
                normalize_mode,         # 17 (radio)
                normalize_mode_column,  # 18 (column visibility)
                use_rollout,            # 19
                rollout_state,          # 20
                loading_msg,            # 21
                inference_done,         # 22
                export_status_box,      # 23
                csv_output,             # 24
                filename_input,         # 25
            ]
        )
        # Run warm start when UI loads
        demo.load(fn=warm_start, inputs=None, outputs=None)

    return demo


# Expose for Spaces & imports
demo = build_ui()

if __name__ == "__main__":
    demo.launch(
        server_name=os.getenv("GRADIO_SERVER_NAME", "127.0.0.1"),
        server_port=int(os.getenv("GRADIO_SERVER_PORT", "7860")),
        show_error=True,
    )
