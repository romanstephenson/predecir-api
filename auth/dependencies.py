from auth.auth_handler import validate_token_with_iam
from fastapi import Depends
from fastapi.security import OAuth2PasswordBearer
from utils.logging_config import setup_logger
from core.config import settings
from fastapi import Depends, HTTPException,status

logger = setup_logger(__name__)

oauth2_scheme = OAuth2PasswordBearer(settings.TOKEN_URL)

async def get_current_user_context(token: str = Depends(oauth2_scheme)) -> dict:
    try:
        user = await validate_token_with_iam(token)

        if user:
            logger.debug(f"Authenticated IAM user: {user.get('username')}")
            return await validate_token_with_iam(token)
        else:
            logger.exception(f"No user with provided token was found to be active")
    except HTTPException:
        raise
    except Exception:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Could not get current user context to verify/validate authentication")