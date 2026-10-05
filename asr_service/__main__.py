"""Start the separate GPU host without access logs containing request data."""
import uvicorn
from .app import create_app
from backend.logging_config import configure_logging
configure_logging()
uvicorn.run(create_app(),host="0.0.0.0",port=9002,access_log=False)
