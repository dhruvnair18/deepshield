# api/gradcam.py
import os
import numpy as np
from PIL import Image
import torch
from torchvision import transforms

DEVICE = os.getenv("DEVICE", "cpu")

def _preproc(pil, size=224):
    tfm = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
    ])
    return tfm(pil).unsqueeze(0)

def _find_last_conv(model: torch.nn.Module):
    last = None
    for m in model.modules():
        if isinstance(m, torch.nn.Conv2d):
            last = m
    if last is None:
        raise RuntimeError("No Conv2d layer found for Grad-CAM.")
    return last

def _to_numpy_img(pil: Image.Image):
    arr = np.asarray(pil.convert("RGB")).astype(np.float32) / 255.0
    return arr  # HWC, [0,1]

def _overlay_heatmap(base_rgb01: np.ndarray, cam01: np.ndarray, alpha=0.45):
    # Try OpenCV colormap if available
    try:
        import cv2
        heat = (cam01 * 255.0).astype(np.uint8)
        heat = cv2.applyColorMap(heat, cv2.COLORMAP_JET)       # HxWxBGR
        heat = cv2.cvtColor(heat, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        out = (1.0 - alpha) * base_rgb01 + alpha * heat
    except Exception:
        # Fallback: red overlay
        heat = np.stack([cam01, np.zeros_like(cam01), np.zeros_like(cam01)], axis=-1)
        out = (1.0 - alpha) * base_rgb01 + alpha * heat
    out = np.clip(out * 255.0, 0, 255).astype(np.uint8)
    return Image.fromarray(out)

@torch.no_grad()
def _forward_logits(model, x):
    return model(x)  # expects single-logit (B,1) or (B,)

def gradcam_overlay(pil_img: Image.Image, model: torch.nn.Module, target_index: int = 0) -> Image.Image:
    """
    Minimal Grad-CAM:
    - Finds last Conv2d
    - Captures activations + grads with hooks
    - Backprop from selected logit (index 0 for single-logit heads)
    Returns a PIL image with heatmap overlay.
    """
    model.eval().to(DEVICE)
    x = _preproc(pil_img).to(DEVICE)

    target_layer = _find_last_conv(model)
    activations = []
    gradients = []

    def fwd_hook(_, __, out):
        activations.append(out)            # (B, C, H, W)

    def bwd_hook(_, grad_in, grad_out):
        gradients.append(grad_out[0])      # (B, C, H, W)

    h1 = target_layer.register_forward_hook(fwd_hook)
    h2 = target_layer.register_full_backward_hook(bwd_hook)

    # Forward
    x.requires_grad_(True)
    logits = model(x)                      # (B,1) or (B,)
    if logits.ndim == 2:
        score = logits[:, target_index].sum()
    else:
        # single-logit head -> use that logit
        score = logits.view(-1).sum()

    # Backward (enable grad just for this pass)
    model.zero_grad(set_to_none=True)
    score.backward(retain_graph=True)

    # Extract tensors
    A = activations[-1].detach()           # (1, C, H, W)
    dA = gradients[-1].detach()            # (1, C, H, W)

    # Weights = mean over spatial dims
    weights = dA.mean(dim=(2,3), keepdim=True)   # (1, C, 1, 1)
    cam = (weights * A).sum(dim=1, keepdim=False)  # (1, H, W)
    cam = torch.relu(cam)
    cam = cam / (cam.max() + 1e-8)         # normalize to [0,1]
    cam01 = cam[0].cpu().numpy()

    # Resize CAM to input image size
    H, W = pil_img.size[1], pil_img.size[0]
    cam01 = Image.fromarray((cam01 * 255).astype(np.uint8)).resize((W, H), Image.BILINEAR)
    cam01 = np.asarray(cam01).astype(np.float32) / 255.0

    base = _to_numpy_img(pil_img)
    overlay = _overlay_heatmap(base, cam01, alpha=0.45)

    # Clean hooks
    h1.remove(); h2.remove()
    return overlay
