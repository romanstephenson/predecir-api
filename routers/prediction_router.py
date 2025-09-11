from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.security import OAuth2PasswordBearer
from entity.recurrence_model import RecurrenceCreate, RecurrenceRecord
from auth.dependencies import get_current_user_context
from core.database import assert_db_alive
from core.config import settings
from utils.logging_config import setup_logger
from fastapi.responses import JSONResponse

logger = setup_logger(__name__)
router = APIRouter()

# Shown only in the API docs so users see how to supply a token
oauth2_scheme = OAuth2PasswordBearer(settings.TOKEN_URL)

@router.post("/predict", status_code=status.HTTP_200_OK)
async def predict( input_data: RecurrenceCreate, request: Request, user: dict = Depends(get_current_user_context), ):
    await assert_db_alive()

    logger.info("Requesting app to load model to app state for ease of use from local directory")
    model = getattr(request.app.state, "model", None)
    
    if model is None:
        logger.error("Model not loaded into app state")
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Prediction model unavailable")
    else:
        logger.info("Model loaded in app state and available for predictions")

    username = user.get("username", "UnknownUser")
    company_id = user.get("company_id")
    logger.info("Prediction requested by %s", username)

    # Predict
    try:
        # Convert request payload to DataFrame with proper column names
        predictDf = input_data.to_dataframe()
        logger.debug("Input as DataFrame:\n%s", predictDf.to_string(index=False))

        # Predict 
        logger.info(f"Generating probabilities based on data in dataframe: { predictDf.to_string(index=False) }")

        probabilities = model.predict_proba( predictDf )[0]

        logger.info(f"Probabilities generated: {probabilities}")
        
        prediction = int( probabilities[1] >= 0.5 )

    except HTTPException:
        raise
    except Exception:
        logger.exception("Model prediction failed")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Prediction model error")

    pred_label = "Recurrence" if prediction == 1 else "No Recurrence"
    logger.info("Prediction: %s | Probability: %.4f", pred_label, probabilities[1] )

    try:
        # 3️ Persist result (fire-and-forget)
        await RecurrenceRecord.create_recurrence(
            recurrence=input_data,
            username=username,
            prediction=pred_label,
            probability=float( probabilities[1] ),
            company_id=company_id,
        )
    except Exception:
        logger.warning(f"Prediction made but record persistence failed")
        status_code = status.HTTP_200_OK

    response_body = {
        "prediction": pred_label,
        "probabilities": {
            "no_recurrence": round( float( probabilities[0] ), 4),
            "recurrence": round( float( probabilities[1] ), 4),
        },
    }

    return JSONResponse(content=response_body, status_code=status_code)

@router.get("/meta")
async def get_model_metadata(request: Request, user: dict = Depends(get_current_user_context)):
    """
    Simple endpoint to expose model version & expected feature columns.
    """
    model = getattr(request.app.state, "model", None)
    if model is None:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Model not loaded")

    try:
        columns = list(model.feature_names_in_)
    except HTTPException:
        raise    
    except AttributeError:
        columns = []  # fallback if unavailable

    return {
        "model_version": settings.MODEL_VERSION,
        "expected_columns": columns,
    }


@router.get("/history")
async def get_user_prediction_history(user: dict = Depends(get_current_user_context)):
    
    await assert_db_alive()
    
    try:
        username = user["username"]
        company_id = user.get("company_id")
        logger.info("Fetching prediction history for %s", username)

        records = await RecurrenceRecord.get_recurrence(username=username, company_id=company_id)
        
    except HTTPException:
        raise
    except Exception:
        logger.exception("Prediction history unavailable")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Prediction history unavailable")
    
    return [r.model_dump() for r in records]