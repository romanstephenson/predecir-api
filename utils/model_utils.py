import os
import urllib.request
from fastapi import HTTPException
from urllib.parse import urlparse, unquote
from utils.logging_config import setup_logger
from core.config import settings
from fastapi import Request, HTTPException
import joblib

logger = setup_logger(__name__)

# def get_model_assets(request: Request):
#     """Retrieve model and encoders from FastAPI app state or raise 503."""
#     model = getattr(request.app.state, "model", None)
#     encoders = getattr(request.app.state, "encoders", None)

#     if model is None or encoders is None:
#         logger.error("Model or encoders not loaded in app state")
#         raise HTTPException(
#             status_code=503,
#             detail="Prediction service not ready. Please try again shortly.",
#         )

#     return model, encoders


def resolve_model_path(model_reference: str, version: str = settings.MODEL_VERSION) -> str:
    """
      Look for the model in model/<version>/.
      If model_reference is a URL (starts with http/https) download & cache.
      Otherwise raise if file missing.
    """
    logger.info("Resolving model path for %s", model_reference)

    local_dir = os.path.join(settings.MODEL_LOCATION, version)   # model/v1

    logger.info("making local directory for model storage")
    os.makedirs(local_dir, exist_ok=True)
    logger.info("directory made")

    # ── remote URL branch ──────────────────────────────
    if model_reference.startswith(("http://", "https://")):
        filename = unquote(os.path.basename(urlparse(model_reference).path))
        local_path = os.path.join(local_dir, filename)

        # always re-download
        if os.path.isfile(local_path):
            logger.info(f"removing old model for redownload from:{model_reference}" )
            os.remove(local_path)

        try:
            logger.info(f"Downloading model to: {local_path} from {model_reference}")
            urllib.request.urlretrieve(model_reference, local_path)
            logger.info("Download complete")

        except Exception as e:
            logger.exception("Download failed")
            raise HTTPException(status_code=500, detail=f"Could not download model from {model_reference}") from e

        return local_path

    # ── filename / relative path branch ─────────────────
    #  first try as absolute path the user gave
    if os.path.isfile(model_reference):
        return model_reference

    #  otherwise join with model/<version>/model file
    full_local_path = os.path.join(local_dir, os.path.basename(model_reference))

    if not os.path.isfile(full_local_path):
        logger.error("Model not found in image: %s", full_local_path)
        raise HTTPException(status_code=500,detail=f"Model file not found in image: {full_local_path}")

    return full_local_path



def load_pipeline_model(model_filename: str, version: str = settings.MODEL_VERSION):
    """
    Resolves and loads a scikit-learn pipeline model saved via joblib.
    
    Parameters:
    - model_filename: str – the name of the model file (e.g. BreastCancerRecurrence_Predecir_2.pkl)
    - version: str – version subfolder under model/ (default from settings)
    
    Returns:
    - Loaded sklearn pipeline
    """
    model_path = resolve_model_path(model_filename, version)
    logger.info(f"Loading model from: {model_path}", )

    try:
        model = joblib.load(model_path)
        return model
    except Exception as e:
        logger.exception("Failed to load model from: %s", model_path)
        raise HTTPException(status_code=500, detail="Failed to load model.")

# def resolve_encoder_files(urls: list, version: str = settings.MODEL_VERSION) -> dict:
#     """
#     Resolves encoder files: uses existing files in container if found,
#     else downloads them only if full URLs are provided.
#     """
#     local_dir = os.path.join(settings.MODEL_LOCATION, version, "encoders")
#     os.makedirs(local_dir, exist_ok=True)

#     local_paths = {}
#     for ref in urls:
#         parsed = urlparse(ref)
#         filename = unquote(os.path.basename(parsed.path)) if parsed.scheme else ref
#         dest_path = os.path.join(local_dir, filename)

#         # Case 1: File already exists locally
#         if os.path.isfile(dest_path):
#             logger.info(f"Encoder found locally: {filename}")
#             local_paths[filename] = dest_path
#             continue

#         # Case 2: Valid URL, attempt to download
#         if parsed.scheme:
#             try:
#                 logger.info(f"Downloading encoder {filename} from {ref}...")
#                 urllib.request.urlretrieve(ref, dest_path)
#                 logger.info(f"Downloaded encoder to {dest_path}")
#                 local_paths[filename] = dest_path
#                 continue
#             except Exception as e:
#                 logger.error(f"Failed to download encoder {filename}: {e}")
#                 raise HTTPException(status_code=500, detail=f"Failed to download encoder: {filename}")

#         # Case 3: Filename given, but file missing in image
#         logger.error(f"Missing encoder file: {dest_path}")
#         raise HTTPException(status_code=500, detail=f"Encoder file not found in image: {filename}")

#     return local_paths
