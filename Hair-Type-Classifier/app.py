import os
import sys
import inspect
import pathlib
from io import BytesIO
from typing import List, Dict, Any

import numpy as np
from PIL import Image

# NEW: env + HTTP client for weather API
from dotenv import load_dotenv
import requests

# Load environment variables from .env (if present)
load_dotenv()
WEATHER_API_KEY = os.getenv("WEATHER_API_KEY")

# ===== Albucore / Albumentations compatibility shim =====
# Some environments install a version of `albucore`
# that does not define `preserve_channel_dim`, but the
# `albumentations` version used when we trained the model
# still expects it to exist. We recreate it here BEFORE
# importing albumentations so that the import does not fail.
try:
    import albucore.utils as _acu
    from typing import Callable
    import numpy as _np

    if not hasattr(_acu, "preserve_channel_dim"):

        def preserve_channel_dim(
            func: Callable[..., _np.ndarray]
        ) -> Callable[..., _np.ndarray]:
            """
            Backwards-compat shim for old Albumentations code that expects
            `albucore.utils.preserve_channel_dim`. Newer albucore removed it,
            so we provide a tiny version that keeps the image channel dims.
            """

            def wrapper(image: _np.ndarray, *args, **kwargs) -> _np.ndarray:
                orig_ndim = image.ndim
                result = func(image, *args, **kwargs)
                # If the number of dimensions changed, try to restore channels
                if result.ndim == orig_ndim:
                    return result
                if orig_ndim == 3 and result.ndim == 2:
                    # e.g. (H, W, C) -> (H, W); put channels back
                    return result[..., None]
                return result

            return wrapper

        # Attach our shim to albucore.utils so Albumentations can import it
        _acu.preserve_channel_dim = preserve_channel_dim  # type: ignore[attr-defined]
except Exception:
    # If albucore isn't present yet during install, just ignore.
    # When it is finally imported the above will already be defined.
    pass

# NOW it is safe to import albumentations
import albumentations as A
from fastai.vision.all import load_learner, PILImage, RandTransform

from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware

# =========================================================
# 0) Windows PosixPath pickle compatibility
# =========================================================

if os.name == "nt":
    # Map PosixPath objects from pickle to WindowsPath
    pathlib.PosixPath = pathlib.WindowsPath  # type: ignore
    print("[Patch] Mapped pathlib.PosixPath -> WindowsPath for pickle loading on Windows.")

# =========================================================
# 1) Albumentations compatibility patches
# =========================================================

def _ensure_fastai_albu_compat(cls):
    """
    Ensure attributes/methods expected by the pickled fastai+albumentations pipeline exist.
    Implemented as safe no-ops/defaults so newer Albumentations works with older pickles.
    """
    # apply_to_images: legacy API some pickles expect
    if not hasattr(cls, "apply_to_images"):
        def _apply_to_images(self, *args, **kwargs):
            return self(*args, **kwargs)
        setattr(cls, "apply_to_images", _apply_to_images)

    # replay_mode: flag used internally in older versions
    if not hasattr(cls, "replay_mode"):
        setattr(cls, "replay_mode", False)

    # deterministic: also touched in some older flows
    if not hasattr(cls, "deterministic"):
        setattr(cls, "deterministic", False)


try:
    from albumentations.core.transforms_interface import BasicTransform

    for name in dir(A):
        obj = getattr(A, name)
        if inspect.isclass(obj) and issubclass(obj, BasicTransform):
            _ensure_fastai_albu_compat(obj)
except Exception:
    # Fallback: best-effort patch for anything callable
    for name in dir(A):
        obj = getattr(A, name)
        if inspect.isclass(obj) and hasattr(obj, "__call__"):
            _ensure_fastai_albu_compat(obj)

# Also patch Compose to always have additional_targets
if hasattr(A, "Compose"):
    Compose = A.Compose
    if not hasattr(Compose, "additional_targets"):
        Compose.additional_targets = {}

    _old_init = Compose.__init__

    def _new_init(self, *args, **kwargs):
        if "additional_targets" not in kwargs:
            kwargs["additional_targets"] = {}
        _old_init(self, *args, **kwargs)

    Compose.__init__ = _new_init  # type: ignore
    print("[Patch] Added default additional_targets = {} to Albumentations Compose.")


# =========================================================
# 1b) AlbumentationsTransform stub for unpickling
# =========================================================

class AlbumentationsTransform(RandTransform):
    """
    Minimal compatible version of the AlbumentationsTransform class
    that was used when training/saving the Learner.

    It's only needed so `load_learner` can find this symbol during unpickling.
    We don't rely on it explicitly in this serving script.
    """
    split_idx, order = None, 2

    def __init__(self, train_aug=None, valid_aug=None):
        super().__init__()
        self.train_aug = train_aug
        self.valid_aug = valid_aug
        self.idx = 0

    def before_call(self, b, split_idx):
        self.idx = split_idx

    def encodes(self, img: PILImage):
        # Defensive: if aug is missing, just return the image unchanged
        if self.idx == 0 and self.train_aug is not None:
            aug_img = self.train_aug(image=np.array(img))["image"]
            return PILImage.create(aug_img)
        if self.idx == 1 and self.valid_aug is not None:
            aug_img = self.valid_aug(image=np.array(img))["image"]
            return PILImage.create(aug_img)
        return img


# 🔑 IMPORTANT: register stub on __main__ so torch.load can find it
sys.modules["__main__"].AlbumentationsTransform = AlbumentationsTransform


# =========================================================
# 2) Load model
# =========================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "hair-resnet18-model.pkl")

if not os.path.isfile(MODEL_PATH):
    raise FileNotFoundError(
        f"Model file not found at {MODEL_PATH}. "
        f"Make sure 'hair-resnet18-model.pkl' is inside the 'models' folder."
    )

print(f"[Info] Loading model from: {MODEL_PATH}")
learn = load_learner(MODEL_PATH)
print("[Info] Model loaded successfully.")

HAIR_LABELS: List[str] = list(map(str, learn.dls.vocab))


# =========================================================
# 3) Simple product catalog & recommender
# =========================================================

PRODUCT_CATALOG: List[Dict[str, Any]] = [
    {
        "name": "Afri’Pure Shea Butter + Marula Moisturising Hair Oil",
        "brand": "Afri’Pure",
        "hair_types": ["coily", "kinky", "curly"],
        "description": "Seals in moisture, softens tight textures, and fights breakage.",
        "image_url": "/products/shea-butter.jpg",
        "buy_url": "",
        "actives": ["Shea Butter", "Marula Oil"],
    },
    {
        "name": "Native Child Castor Oil – Hairgrowth Oil",
        "brand": "Native Child",
        "hair_types": ["coily", "kinky", "curly"],
        "description": "Ricinoleic-acid rich oil to reduce breakage and support growth.",
        "image_url": "/products/castor-oil.jpg",
        "buy_url": "",
        "actives": ["Castor Oil"],
    },
    {
        "name": "Afri’Pure Vegetable Glycerine (100% Pure)",
        "brand": "Afri’Pure",
        "hair_types": ["wavy", "curly", "coily", "kinky"],
        "description": "Humectant that draws moisture into hair—great with a sealing oil.",
        "image_url": "/products/glycerin.jpg",
        "buy_url": "",
        "actives": ["Glycerin"],
    },
    {
        "name": "Pure Hydrolyzed Collagen (Peptide Powder)",
        "brand": "Collagen Co.",
        "hair_types": ["straight", "wavy", "curly", "coily", "kinky"],
        "description": "Small peptide fragments that can strengthen hair over time.",
        "image_url": "/products/hydrolyzed-protein.jpg",
        "buy_url": "",
        "actives": ["Hydrolyzed Protein"],
    },
    {
        "name": "Afri’Pure Elixir – Shea Butter + Marula",
        "brand": "Afri’Pure",
        "hair_types": ["coily", "kinky", "curly"],
        "description": "Rich leave-in oil blend to reduce frizz and protect ends.",
        "image_url": "/products/marula-oil.jpg",
        "buy_url": "",
        "actives": ["Marula Oil", "Shea Butter"],
    },
    {
        "name": "Afri’Pure Argan Oil – Hydrating Hair Oil",
        "brand": "Afri’Pure",
        "hair_types": ["wavy", "curly", "straight"],
        "description": "Lightweight shine and softness without heavy build-up.",
        "image_url": "/products/argan-oil.jpg",
        "buy_url": "",
        "actives": ["Argan Oil", "Vitamin E"],
    },
    {
        "name": "Afri’Pure Jojoba Oil – Balancing Hair Oil",
        "brand": "Afri’Pure",
        "hair_types": ["straight", "wavy", "curly"],
        "description": "Wax-ester oil that balances scalp oils and reduces inflammation.",
        "image_url": "/products/jojoba-oil.jpg",
        "buy_url": "",
        "actives": ["Jojoba Oil"],
    },
]


def recommend_products(hair_probs: Dict[str, float], top_k: int = 4) -> List[Dict[str, Any]]:
    # Dominant predicted hair type
    best_label = max(hair_probs, key=hair_probs.get)
    best_prob = hair_probs[best_label]

    recs: List[Dict[str, Any]] = []
    for p in PRODUCT_CATALOG:
        targets = [t.lower() for t in p["hair_types"]]
        if best_label.lower() in targets:
            score = best_prob * 100  # strong match
        else:
            score = best_prob * 45   # soft match

        recs.append({
            **p,
            "match_score": round(float(score), 1),
            "for_label": best_label,
        })

    recs.sort(key=lambda x: x["match_score"], reverse=True)
    return recs[:top_k]


# =========================================================
# 4) FastAPI app with CORS for React frontend
# =========================================================

app = FastAPI(title="Trichofy Hair API")

# Allow localhost + optional deployed frontend origin from env
frontend_origin = os.getenv("FRONTEND_ORIGIN")  # e.g. https://trichofy.vercel.app
origins = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    "https://trichofy-frontend-ymh1.onrender.com",
    "https://www.trichofy.co.za",
    "https://trichofy.co.za",
]
if frontend_origin:
    origins.append(frontend_origin)

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Simple health / root check
@app.get("/")
def root():
    return {"status": "ok", "message": "Trichofy Hair API is running."}


def _predict_from_pil(img: Image.Image):
    pred, pred_idx, probs = learn.predict(PILImage.create(np.array(img)))
    probs_dict = {HAIR_LABELS[i]: float(probs[i]) for i in range(len(HAIR_LABELS))}
    products = recommend_products(probs_dict)
    return str(pred), probs_dict, products


@app.post("/predict")
async def predict_endpoint(file: UploadFile = File(...)):
    """
    Accepts an uploaded image and returns:
    - predicted hair type
    - probabilities per class
    - recommended products with match scores
    """
    try:
        contents = await file.read()
        img = Image.open(BytesIO(contents)).convert("RGB")
    except Exception:
        return {"error": "Invalid image file."}

    pred_label, probs_dict, products = _predict_from_pil(img)

    return {
        "hair_type": pred_label,
        "probabilities": probs_dict,
        "products": products,
    }


# =========================================================
# 5) Weather endpoint for Seasonal Hair Adjustments
# =========================================================

@app.get("/weather")
def get_weather(city: str = "Johannesburg", country: str = "ZA"):
    """
    Fetch current weather for a given South African city using OpenWeatherMap.

    Default: Johannesburg, ZA
    """
    if not WEATHER_API_KEY:
        return {"error": "Weather API key not configured on the server."}

    url = "https://api.openweathermap.org/data/2.5/weather"
    params = {
        "q": f"{city},{country}",
        "appid": WEATHER_API_KEY,
        "units": "metric",
    }

    try:
        response = requests.get(url, params=params, timeout=8)
    except Exception as e:
        return {"error": "Failed to reach weather service.", "details": str(e)}

    if response.status_code != 200:
        return {
            "error": "Weather API returned an error.",
            "status_code": response.status_code,
            "details": response.text,
        }

    data = response.json()

    weather_info = {
        "city": data.get("name"),
        "temp": data["main"]["temp"],
        "feels_like": data["main"]["feels_like"],
        "humidity": data["main"]["humidity"],
        "condition": data["weather"][0]["main"],
        "description": data["weather"][0]["description"],
        "icon": data["weather"][0]["icon"],
    }

    return weather_info


if __name__ == "__main__":
    import uvicorn

    # Usually PORT is injected in env on hosting providers
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run(app, host="0.0.0.0", port=port)
