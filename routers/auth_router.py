import httpx
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from auth.auth_handler import validate_token_with_iam
from utils.logging_config import setup_logger
from core.config import settings

logger = setup_logger(__name__)

router = APIRouter()

# JWT dependency
# It's used only for API docs to show where a token can be obtained
oauth2_scheme = OAuth2PasswordBearer(settings.TOKEN_URL)

# IAM login
IAM_API_TOKEN_URL = settings.IAM_API_TOKEN_URL

# IAM logout URL
IAM_API_LOGOUT_URL = settings.IAM_API_LOGOUT_URL
 
@router.post("/tokenlogin")
async def login(form_data: OAuth2PasswordRequestForm = Depends()):

    logger.info(f"Received login request for: {form_data.username}")

    try:
        logger.info(f"Attempting to contact IAM API at: {IAM_API_TOKEN_URL}")
        
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.post(
                IAM_API_TOKEN_URL,
                data={"username": form_data.username, "password": form_data.password},
                headers={"Content-Type": "application/x-www-form-urlencoded"},
            )

        if response.status_code == 200:
            logger.info(f"Login successful for user: {form_data.username}")
            return response.json()
        else:
            logger.error(
                f"IAM login failed for user: {form_data.username} | "
                f"Status Code: {response.status_code} | Response: {response.text}"
            )
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail=f"IAM login failed: {response.json().get('detail', 'Unknown error')}"
            )

    except httpx.ConnectError:
        logger.exception("Failed to connect to IAM API")
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="IAM service unavailable. Please try again later.")

    except httpx.ReadTimeout:
        logger.exception("IAM API request timed out")
        raise HTTPException(status_code=status.HTTP_504_GATEWAY_TIMEOUT, detail="IAM service timeout. Please try again later.")

    except httpx.RequestError as e:
        logger.exception(f"HTTP error occurred while contacting IAM API: {str(e)}")
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail="Error communicating with IAM service.")

    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Unexpected login error: {str(e)}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Unexpected error during login.")


@router.post("/logout")
async def logout(token: str = Depends(oauth2_scheme)):
    try:
        logger.info(f"Logout requested for user: {token.capitalize}")

        # Step 1: Validate token locally or via IAM
        try:
            logger.debug("Validating token before logout")
            
            await validate_token_with_iam(token)

            logger.debug("Token validated successfully")

        except HTTPException as e:
            
            logger.warning(f"Token validation failed: {e.detail}")
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid or expired token")

        # Step 2: Call IAM logout
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    IAM_API_LOGOUT_URL,
                    headers={"Authorization": f"Bearer {token}"}
                )
            if response.status_code == 200:
                logger.info("Token logout successful with IAM")
                return response.json()
            else:
                logger.error(
                    f"IAM logout failed | Status: {response.status_code} | Detail: {response.text}"
                )
                raise HTTPException(
                    status_code=response.status_code,
                    detail=f"IAM logout failed: {response.text}"
                )
        except httpx.RequestError as e:
            logger.exception("HTTP request to IAM failed")
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY, detail="Failed to communicate with IAM service for logout"
            )

    except HTTPException:
        raise  # Rethrow cleanly for FastAPI to handle

    except Exception as e:
        logger.exception(f"Unexpected error during logout: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Unexpected server error during logout"
        )