import asyncio
from motor.motor_asyncio import AsyncIOMotorClient
from core.config import settings
from fastapi import HTTPException
from utils.logging_config import setup_logger

logger = setup_logger(__name__)

MONGO_URL = settings.MONGO_URL
DB_NAME = settings.MONGO_DB_NAME

MAX_RETRIES = int(settings.MONGO_MAX_RETRIES)
RETRY_DELAY = int(settings.MONGO_RETRY_DELAY)  # seconds

client = AsyncIOMotorClient(MONGO_URL, serverSelectionTimeoutMS=3000)
db = client[DB_NAME]

# Health check utility
async def assert_db_alive():
    try:
        logger.info(f"Mongo URI being used in assert db alive: {settings.MONGO_URL}")
        await db.command("ping")
    except Exception as e:
        logger.error("MongoDB ping failed: %s", str(e))
        raise HTTPException(status_code=503, detail="Database connection unavailable")

# Safe retry-based startup connection check
async def verify_db_connection():
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            await assert_db_alive()
            logger.info(f"Mongo URI being used verify db connection: {settings.MONGO_URL}")

            logger.info("MongoDB connection verified on attempt %d", attempt)
            return
        except HTTPException as e:
            logger.warning("MongoDB connection attempt %d failed: %s", attempt, e.detail)
            if attempt < MAX_RETRIES:
                await asyncio.sleep(RETRY_DELAY)
            else:
                logger.critical("Could not connect to MongoDB after %d attempts. Shutting down.", MAX_RETRIES)
                raise SystemExit("Failed to connect to MongoDB")
