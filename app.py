from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from routers.prediction_router import router as predict_router
from routers.auth_router import router as auth_router
from dotenv import load_dotenv
from utils.logging_config import setup_logger
from core import database as db  #triggers connection check on app startup
from core.config import settings
from contextlib import asynccontextmanager
from utils.model_utils import load_pipeline_model

load_dotenv()

logger = setup_logger(__name__)
logger.info("Predecir API initializing....")

# Confirm DB connection is present
logger.info("Initializing Database")

@asynccontextmanager
async def lifespan(app: FastAPI):
    await db.verify_db_connection()
    logger.info("Database verified.")

    logger.info("Resolving and loading model and encoders...")
    try:
        #model_path = resolve_model_path(settings.RECUR_MODEL_FILENAME, settings.MODEL_VERSION)

        # load model
        model = load_pipeline_model(settings.RECUR_MODEL_FILENAME, settings.MODEL_VERSION)

        # resolve encoder files (get paths)
        # encoder_paths = resolve_encoder_files(
        #     settings.encoder_url_list(),
        #     settings.MODEL_VERSION
        # )

        # load encoders into memory
        # encoders = {}
        # for name, path in encoder_paths.items():
        #     with open(path, "rb") as f:
        #         encoders[name] = pickle.load(f)

        # store in app state
        app.state.model = model
        logger.info("Model downloaded and will now be stored in the app state for ease of reference")
        #app.state.encoders = encoders

        logger.info("Model successfully loaded.")
    except Exception:
        logger.exception("Failed to load models or encoders.")
        raise 

    yield

origins = settings.ALLOWED_ORIGINS

app = FastAPI(
        title=settings.APP_NAME,
        version=settings.APP_VERSION,
        lifespan=lifespan
        )
logger.info("DB Initialization complete.")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register routes
logger.info("Registering routes")
app.include_router(auth_router, prefix="/predecir/auth")
app.include_router(predict_router, prefix="/predecir")
logger.info("Routes registered")


# Root endpoint
@app.get("/")
def root():
    logger.info(f"{settings.APP_NAME} is running")
    return {"message": f"{settings.APP_NAME} is running"}

logger.info("FastAPI app initializing completed")
