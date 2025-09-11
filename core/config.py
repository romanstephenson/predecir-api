from pydantic_settings import BaseSettings
from typing import List
import os

class Settings(BaseSettings):
    
    # Model settings
    MODEL_VERSION : str = os.getenv("MODEL_VERSION", "v1")
    MODEL_LOCATION: str = os.getenv("MODEL_LOCATION")
    # ENCODER_URLS: str = os.getenv("ENCODER_URLS")
    RECUR_MODEL_FILENAME: str = os.getenv("RECUR_MODEL_FILENAME")
    # def encoder_url_list(self) -> List[str]:
    #     return [url.strip() for url in self.ENCODER_URLS.split(",")]

    # Security and auth
    JWT_SECRET_KEY: str = os.getenv("JWT_SECRET_KEY","<putyourkeyhere>")
    ACCESS_TOKEN_EXPIRE_MINUTES: int = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES",60) )
    ALGORITHM: str = os.getenv("ALGORITHM", "HS256")

    # CORS and logging
    ALLOWED_ORIGINS: List[str] = os.getenv("ALLOWED_ORIGINS", "http://localhost").strip("[]").replace("\"", "").split(", ")
    LOG_DIR: str = os.getenv("LOG_DIR","logs" )
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "DEBUG") # Options: DEBUG, INFO, WARNING, ERROR
    LOG_FILE_NAME : str = os.getenv("LOG_FILE_NAME","predecirapp.log")
    LOG_BACKUP_COUNT : int = int(os.getenv("LOG_BACKUP_COUNT", 3))
    LOG_MAX_SIZE_MB : int = int(os.getenv("LOG_MAX_SIZE_MB", 5))

    # MongoDB
    MONGO_URL: str = os.getenv("MONGO_URL", "mongodb://localhost:27017")
    MONGO_DB_NAME: str = os.getenv("MONGO_DB", "oncovista")
    MONGO_MAX_RETRIES: int = os.getenv("MONGO_MAX_RETRIES", 5)
    MONGO_RETRY_DELAY: int = os.getenv("MONGO_RETRY_DELAY", 3)

    # App Info
    DEBUG: bool = os.getenv("DEBUG", "True").lower() == "true"
    ENV: str = os.getenv("ENV", "development")
    APP_NAME: str = os.getenv("APP_NAME", "PREDECIR API - Breast Cancer Recurrence Predictor")
    APP_VERSION: str = os.getenv("APP_VERSION", "1.0")
    APP_PORT: int = int(os.getenv("APP_PORT", 8001 ) )
    APP_HOST: str = os.getenv("APP_HOST", "0.0.0.0" )

    # IAM API
    TOKEN_URL : str = os.getenv("TOKEN_URL", "/predecir/auth/tokenlogin")
    IAM_API_TOKEN_URL : str = os.getenv("IAM_API_TOKEN_URL", "http://localhost:8000/predecir/auth/tokenlogin")
    IAM_API_VALIDATE_URL : str = os.getenv("IAM_API_VALIDATE_URL", "http://localhost:8000/predecir/auth/validate")
    IAM_API_LOGOUT_URL : str = os.getenv("IAM_API_LOGOUT_URL", "http://localhost:8000/predecir/auth/logout")


    class Config:
        # Dynamically load env file based on ENV value
        env_file = (
            ".env"
            if os.getenv("ENV", "development").lower() == "production"
            else ".env"
        )
        case_sensitive = True

# Export instance to use across app
settings = Settings()
