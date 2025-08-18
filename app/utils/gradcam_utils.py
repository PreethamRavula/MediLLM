import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


def register_hooks(model):
    activations = {}
    gradients = {}

    def forward_hook(module, input, output):
        activations["value"] = output

    def backward_hook(module, grad_input, grad_output):
        gradients["value"] = grad_output[0]

    layer = model.image_encoder.layer4
    fwd_handle = layer.register_forward_hook(forward_hook)
    bwd_handle = layer.register_full_backward_hook(backward_hook)

    return activations, gradients, fwd_handle, bwd_handle


def generate_gradcam(image_pil, activations, gradients):
    grads = gradients["value"]
    acts = activations["value"]

    # Out-of-place Grad-CAM weighting
    pooled_grads = torch.mean(grads, dim=[0, 2, 3])
    for i in range(acts.shape[1]):
        acts[:, i, :, :] *= pooled_grads[i]

    # Normalize heatmap
    heatmap = torch.mean(acts, dim=1).squeeze().detach().cpu().numpy()
    heatmap = np.maximum(heatmap, 0)
    heatmap /= heatmap.max() + 1e-8

    # Convert to image and overlay
    heatmap_resized = Image.fromarray(np.uint8(255 * heatmap)).resize((224, 224))
    heatmap_array = np.array(heatmap_resized)
    colormap = plt.cm.jet(heatmap_array / 255.0)[..., :3]  # shape (H, W, 3), RGB

    # Combine with original image
    image_np = np.array(image_pil.resize((224, 224)).convert("RGB")) / 255.0
    overlay = (0.6 * image_np + 0.4 * colormap) * 255
    overlay = overlay.astype(np.uint8)

    return Image.fromarray(overlay)
