# api/model_loader.py
import os
#from functools import lru_cache

# Expect these globals in your module (keep your existing ones if already defined)
WEIGHTS_PATH     = os.getenv("WEIGHTS_PATH", "models/deepfake_mnv2.pth")
MODEL_FRAMEWORK  = os.getenv("MODEL_FRAMEWORK", "pytorch").lower()
DEVICE           = os.getenv("DEVICE", "cpu")
FAKE_CLASS_INDEX = int(os.getenv("FAKE_CLASS_INDEX", "1"))
INVERT_PROBS     = os.getenv("INVERT_PROBS", "false").lower() in ("1", "true", "yes")

EFFB0 = os.getenv("MODEL_ARCH", "mobilenet_v2").lower()  # mobilenet_v2 | efficientnet_b0

def _build_effb0(out_features=1):
    import timm
    model = timm.create_model("efficientnet_b0", pretrained=True, num_classes=out_features)
    return model

def _strip_module_prefix(state_dict):
    # Remove 'module.' from keys if saved with DataParallel
    if not isinstance(state_dict, dict):
        return state_dict
    needs_strip = any(k.startswith("module.") for k in state_dict.keys())
    if not needs_strip:
        return state_dict
    return {k.replace("module.", "", 1): v for k, v in state_dict.items()}

def _unwrap_state_dict(maybe_wrapped):
    # Some checkpoints look like {'state_dict': OrderedDict(...)}
    if isinstance(maybe_wrapped, dict) and "state_dict" in maybe_wrapped:
        return maybe_wrapped["state_dict"]
    return maybe_wrapped

def get_model():
    if not os.path.exists(WEIGHTS_PATH):
        raise FileNotFoundError(
            f"Model file not found at {WEIGHTS_PATH}. "
            f"Set WEIGHTS_PATH env var or switch to DEMO_MODE=true."
        )

    if MODEL_FRAMEWORK == "pytorch":
        import torch
        import torch.nn as nn
        from torchvision import models

        import torch
        import torch.nn as nn
        from torchvision import models

        raw_state = torch.load(WEIGHTS_PATH, map_location=DEVICE)

        # --- unwrap state_dict if nested or DataParallel ---
        def _unwrap_state_dict(sd):
            if hasattr(sd, "state_dict"):
                sd = sd.state_dict()
            if isinstance(sd, dict):
                return {k.replace("module.", ""): v for k, v in sd.items()}
            return sd

        state = _unwrap_state_dict(raw_state)

        arch = os.getenv("MODEL_ARCH", "mobilenet_v2").lower()
        model = None
        head_type = "unknown"

        if arch == "efficientnet_b0":
            # Create EfficientNet-B0 model
            from timm import create_model
            out_feats = 1  # default
            if isinstance(state, dict):
                for k in ("classifier.bias", "classifier.weight"):
                    if k in state:
                        out_feats = state[k].shape[0] if k.endswith(".bias") else state[k].shape[0]
                        break
            model = _build_effb0(out_features=out_feats)
            try:
                model.load_state_dict(state, strict=True)
            except Exception:
                model.load_state_dict(state, strict=False)
            head_type = "single" if out_feats == 1 else "two_class"

        else:
            # fallback: mobilenet_v2 (default)
            model = models.mobilenet_v2(weights=None)
            head_key = None
            if isinstance(state, dict):
                for k in ("classifier.1.weight", "classifier.1.bias"):
                    if k in state: head_key = k; break
            if head_key:
                out_feats = state[head_key.replace(".bias",".weight")].shape[0] if head_key.endswith(".bias") else state[head_key].shape[0]
                model.classifier[1] = nn.Linear(model.last_channel, out_feats)
                model.load_state_dict(state, strict=True)
                head_type = "single" if out_feats == 1 else "two_class"
            else:
                model.classifier[1] = nn.Linear(model.last_channel, 2)
                model.load_state_dict(state, strict=False)
                head_type = "two_class"

        model.eval().to(DEVICE)
        meta = {"fake_class_index": FAKE_CLASS_INDEX, "invert": INVERT_PROBS, "head": head_type, "arch": arch}
        return ("pytorch", model, meta)


    elif MODEL_FRAMEWORK == "keras":
        # Keras / TF
        try:
            from tensorflow.keras.models import load_model
            model = load_model(WEIGHTS_PATH)
            # We can’t easily infer head type here; caller will branch on last-dim
            meta = {
                "fake_class_index": FAKE_CLASS_INDEX,
                "invert": INVERT_PROBS,
                "head": "unknown"
            }
            return ("keras", model, meta)
        except Exception as e:
            raise RuntimeError(f"Failed to load Keras model: {e}")

    else:
        raise RuntimeError("Unknown MODEL_FRAMEWORK. Use 'pytorch' or 'keras'.")
