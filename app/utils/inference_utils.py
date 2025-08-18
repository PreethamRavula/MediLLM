import sys
import torch
import yaml
import numpy as np
from pathlib import Path
from transformers import AutoTokenizer
from torchvision import transforms

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(ROOT_DIR))

from src.multimodal_model import MediLLMModel
from app.utils.gradcam_utils import register_hooks, generate_gradcam


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Label map
inv_map = {0: "low", 1: "medium", 2: "high"}

# Tokenizer and image transform
tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
image_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.CenterCrop(224),
    transforms.ToTensor()
])


def load_model(mode, model_path, config_path=str(Path("config/config.yaml").resolve())):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)[mode]

    model = MediLLMModel(
        mode=mode,
        dropout=config["dropout"],
        hidden_dim=config["hidden_dim"]
    )
    state = torch.load(model_path, map_location=DEVICE)
    model.load_state_dict(state)
    model.to(DEVICE)
    model.eval()
    return model


def attention_rollout(attentions, last_k=4, residual_alpha=0.5):
    """
    attentions_tuple: tuple/list of layer attentions; each is (B,H,S,S)
    last_k: only roll back through the last k layers (keeps contrast)
    residual_alpha: how much identity to add before normalizing (preserve token self-info)
    returns: [B, S, S] rollout matrix, or None if input is invalid
    """
    if attentions is None:
        return None
    if isinstance(attentions, (list, tuple)) and len(attentions) == 0:
        return None

    first = attentions[0]
    if first is None or first.ndim != 4:
        return None  # expect [B, H, S, S]

    B, H, S, _ = first.shape
    eye = torch.eye(S, device=first.device).unsqueeze(0).expand(B, S, S)  # [B, S, S]

    L = len(attentions)
    if last_k is None:
        last_k = L
    if last_k <= 0:
        # No layers selected -> return identity (no propagation)
        return eye.clone()

    start = max(0, L - last_k)
    A = None
    for layer in range(start, L):
        a = attentions[layer]
        if a is None or a.ndim != 4 or a.shape[0] != B or a.shape[-1] != S:
            # Skip malformed layer
            continue
        a = a.mean(dim=1)  # [B, S, S] (avg heads)
        a = a + float(residual_alpha) * eye
        a = a / (a.sum(dim=-1, keepdim=True) + 1e-12)  # row-normalize
        A = a if A is None else torch.bmm(A, a)

    # if we never multiplied like when all layers skipped, fall back to identity
    return A if A is not None else eye.clone()  # [B,S,S]


def merge_wordpieces(tokens, scores):
    merged_tokens, merged_scores = [], []
    cur_tok, cur_scores = "", []
    for t, s in zip(tokens, scores):
        if t.startswith("##"):
            cur_tok += t[2:]
            cur_scores.append(s)
        else:
            if cur_tok:
                merged_tokens.append(cur_tok)
                merged_scores.append(sum(cur_scores) / max(1, len(cur_scores)))
            cur_tok, cur_scores = t, [s]
    if cur_tok:
        merged_tokens.append(cur_tok)
        merged_scores.append(sum(cur_scores) / max(1, len(cur_scores)))
    return merged_tokens, merged_scores


def _normalize_for_display_wordlevel(attn_scores, normalize_mode="visual", temperature=0.30):
    """
    Convert raw *word-level* token scores into:
      - probabilistic mode: probabilities that sum to 1.0 (100%), with labels like "0.237 | 23.7% (contrib)"
      - visual mode: min-max + gamma scaling (contrast, not sum-to-100), with labels like "0.68 | visual score"

      Returns:
        attn_final: np.ndarray of floats in [0, 1] for color scale
        labels: list[str] per token (tooltip text; first number stays up front for your color_map bucketing)
    """
    attn_array = np.array(attn_scores, dtype=float)

    if normalize_mode == "probabilistic":
        # ---- percentage view that sums up to 100% ----
        attn_array = np.maximum(attn_array, 0.0)
        if attn_array.max() > 0:
            attn_array = attn_array / (attn_array.max() + 1e-12)  # scale to [0, 1] for stability
        # sharpen (lower temp => peakier)
        attn_array = np.power(attn_array + 1e-12, 1.0 / max(1e-6, float(temperature)))
        prob = attn_array / (attn_array.sum() + 1e-12)
        percent = prob * 100.0

        # keep prob (0..1) for color scale; label with % contrib
        labels = [f"{prob[i]:.3f} | {percent[i]:.1f}% (contrib)" for i in range(len(prob))]
        return prob, labels
    else:
        # ---- visual: min-max + gamma (contrast, not sum-to-100) ---
        if attn_array.max() > attn_array.min():
            attn_array0 = (attn_array - attn_array.min()) / (attn_array.max() - attn_array.min() + 1e-8)
            attn_array0 = np.clip(np.power(attn_array0, 0.75), 0.1, 1.0)
        else:
            attn_array0 = np.zeros_like(attn_array)
        labels = [f"{attn_array0[i]:.2f} | visual score" for i in range(len(attn_array0))]
        return attn_array0, labels


def predict(
    model,
    mode,
    emr_text=None,
    image=None,
    normalize_mode="visual",
    need_token_vis=False,
    use_rollout=False
):
    """
    normalize_mode: "visual" (min-max + gamma boost) or "probabilistic" (softmax)
    need_token_vis: request/compute token-level attentions (Doctor mode + text/multimodal)
    use_rollout: use attention rollout across layers
    """
    input_ids = attention_mask = img_tensor = None
    cam_image = None
    highlighted_tokens = None
    top5 = []

    if mode in ["text", "multimodal"] and emr_text:
        text_tokens = tokenizer(
            emr_text,
            return_tensors="pt",
            truncation=True,
            padding="max_length",
            max_length=128,
        )
        input_ids = text_tokens["input_ids"].to(DEVICE)
        attention_mask = text_tokens["attention_mask"].to(DEVICE)

    if mode in ["image", "multimodal"] and image:
        img_tensor = image_transform(image).unsqueeze(0).to(DEVICE)

    # Only Register hooks for Grad-CAM if needed
    if mode in ["image", "multimodal"]:
        activations, gradients, fwd_handle, bwd_handle = register_hooks(model)
        model.zero_grad()

    # === Forward ===
    # Only enable attentions when planning to visualize them
    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        image=img_tensor,
        output_attentions=bool(need_token_vis and (mode in ["text", "multimodal"])),
        return_raw_attentions=bool(use_rollout and need_token_vis)
    )

    logits = outputs["logits"]
    if logits.numel() == 0:
        raise ValueError("Model returned empty logits. Check input format.")

    probs = torch.softmax(logits, dim=1)
    pred = torch.argmax(probs, dim=1).item()
    confidence = probs.squeeze()[pred].item()

    # === Grad-CAM ===
    if mode in ["image", "multimodal"]:
        # Enable gradients only for Grad-CAM
        logits[0, pred].backward(retain_graph=True)
        cam_image = generate_gradcam(image, activations, gradients)
        fwd_handle.remove()
        bwd_handle.remove()

    # === Token-level attention ===
    if need_token_vis and (mode in ["text", "multimodal"]):
        token_attn_scores = None

        if use_rollout and outputs.get("raw_attentions") is not None:
            # partial rollout
            # roll: [B, S, S]; roll[b, 0, :] is CLS-to-all tokens for that batch item
            roll = attention_rollout(outputs["raw_attentions"], last_k=4, residual_alpha=0.5)  # [B,S,S]  # (S, S)
            if roll is not None:
                # roll: [B, S, S]; pick CLS row (index 0)
                cls_to_tokens = roll[0, 0].detach().cpu().numpy().tolist()  # CLS row
                token_attn_scores = cls_to_tokens
        elif outputs.get("token_attentions") is not None:
            token_attn_scores = outputs["token_attentions"].squeeze().tolist()

        if token_attn_scores is not None:
            # Filter out specials/pad + aligh to wordpieces
            ids = input_ids[0].tolist()
            amask = attention_mask[0].tolist() if attention_mask is not None else [1] * len(ids)
            wp_all = tokenizer.convert_ids_to_tokens(ids, skip_special_tokens=False)
            special_ids = set(tokenizer.all_special_ids)
            keep_idx = [i for i, (tid, m) in enumerate(zip(ids, amask)) if (tid not in special_ids) and (m == 1)]
            wp_tokens = [wp_all[i] for i in keep_idx]
            wp_scores = [token_attn_scores[i] if i < len(token_attn_scores) else 0.0 for i in keep_idx]

            # Merge wordpieces into words
            word_tokens, attn_scores = merge_wordpieces(wp_tokens, wp_scores)

            # Build Top-5 (probabilistic normalization for ranking)
            _probs_for_rank, _ = _normalize_for_display_wordlevel(
                attn_scores, normalize_mode="probabilistic", temperature=0.30
            )
            pairs = list(zip(word_tokens, _probs_for_rank))
            pairs.sort(key=lambda x: x[1], reverse=True)
            top5 = [(tok, float(p * 100.0)) for tok, p in pairs[:5]]

            # Final display (probabilistic or visual)
            attn_final, labels = _normalize_for_display_wordlevel(
                attn_scores,
                normalize_mode=normalize_mode,
                temperature=0.30,
            )

            highlighted_tokens = [(tok, labels[i]) for i, tok in enumerate(word_tokens)]

        print("🧪 Normalization Mode Received:", normalize_mode)
        if highlighted_tokens:
            print("🟣 Highlighted tokens sample:", highlighted_tokens[:5])
        else:
            print("🟣 No highlighted tokens (no text or attentions unavailable).")

    return inv_map[pred], cam_image, highlighted_tokens, confidence, probs.tolist(), top5
