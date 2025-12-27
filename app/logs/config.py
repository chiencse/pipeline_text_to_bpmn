"""
PostgreSQL connection configuration for logging.
"""
import os
import logging
from typing import Optional

# Load environment variables from .env file if present
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    # python-dotenv not installed, will use system env vars only
    pass

logger = logging.getLogger(__name__)

# PostgreSQL connection settings
# You can override these with environment variables or .env file
DB_CONFIG = {
    "host": os.getenv("PGHOST", "localhost"),
    "port": int(os.getenv("PGPORT", "5432")),
    "database": os.getenv("PGDATABASE", "langgraph_logs"),
    "user": os.getenv("PGUSER", "postgres"),
    "password": os.getenv("PGPASSWORD", ""),
}

# Debug: Log loaded config (without password)
def _log_config():
    """Log current database configuration (without password) for debugging."""
    config_debug = {k: v if k != "password" else "***" for k, v in DB_CONFIG.items()}
    logger.debug(f"PostgreSQL config loaded: {config_debug}")
    # Also check if env vars are set
    env_vars = ["PGHOST", "PGPORT", "PGDATABASE", "PGUSER", "PGPASSWORD"]
    env_status = {var: "✓" if os.getenv(var) else "✗" for var in env_vars}
    logger.debug(f"Environment variables status: {env_status}")

# Log config on module load
_log_config()

def get_db_connection_string() -> str:
    """
    Returns PostgreSQL connection string in format:
    postgresql://user:password@host:port/database
    """
    config = DB_CONFIG
    return (
        f"postgresql://{config['user']}:{config['password']}"
        f"@{config['host']}:{config['port']}/{config['database']}"
    )

def get_db_config() -> dict:
    """
    Returns database configuration dictionary.
    """
    return DB_CONFIG.copy()

