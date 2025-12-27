"""
PostgreSQL logger for pipeline state logging.
"""
import json
import logging
from datetime import datetime
from typing import Dict, Any, Optional, List

try:
    import psycopg2
    from psycopg2.extras import Json
    from psycopg2.pool import SimpleConnectionPool
    PSYCOPG2_AVAILABLE = True
except ImportError:
    PSYCOPG2_AVAILABLE = False
    logging.warning("psycopg2 not installed. Install with: pip install psycopg2-binary")

from app.logs.config import get_db_config, get_db_connection_string

logger = logging.getLogger(__name__)

# Connection pool (initialized on first use)
_connection_pool: Optional[Any] = None


def init_connection_pool(minconn=1, maxconn=10):
    """
    Initialize PostgreSQL connection pool.
    """
    global _connection_pool
    if not PSYCOPG2_AVAILABLE:
        raise ImportError("psycopg2 is required for PostgreSQL logging. Install with: pip install psycopg2-binary")
    
    if _connection_pool is None:
        try:
            config = get_db_config()
            # Log config for debugging (without password)
            config_debug = {k: v if k != "password" else "***" for k, v in config.items()}
            logger.info(f"Attempting to connect to PostgreSQL: {config_debug}")
            
            _connection_pool = SimpleConnectionPool(
                minconn=minconn,
                maxconn=maxconn,
                host=config["host"],
                port=config["port"],
                database=config["database"],
                user=config["user"],
                password=config["password"]
            )
            logger.info("PostgreSQL connection pool initialized successfully")
        except Exception as e:
            config = get_db_config()
            config_debug = {k: v if k != "password" else "***" for k, v in config.items()}
            logger.error(f"Failed to initialize PostgreSQL connection pool: {e}")
            logger.error(f"Connection config used: {config_debug}")
            logger.error("Please check:")
            logger.error("  1. PostgreSQL server is running")
            logger.error("  2. Environment variables are set correctly (PGHOST, PGPORT, PGDATABASE, PGUSER, PGPASSWORD)")
            logger.error("  3. Database exists and user has permissions")
            raise
    return _connection_pool


def get_connection():
    """
    Get a connection from the pool.
    """
    pool = init_connection_pool()
    return pool.getconn()


def return_connection(conn):
    """
    Return a connection to the pool.
    """
    if _connection_pool:
        _connection_pool.putconn(conn)


def log_state_to_db(state: Dict[str, Any], step: Optional[str] = None, table_name: str = "pipeline_logs"):
    """
    Log pipeline state to PostgreSQL database.
    
    Args:
        state: Pipeline state dictionary
        step: Optional step name (e.g., "pipeline_a", "pipeline_b")
        table_name: Name of the log table (default: "pipeline_logs")
    
    Returns:
        log_id: ID of the inserted log record, or None if failed
    """
    if not PSYCOPG2_AVAILABLE:
        logger.warning("psycopg2 not available, skipping database log")
        return None
    
    conn = None
    try:
        conn = get_connection()
        cursor = conn.cursor()
        
        # Convert state to JSONB
        state_json = Json(state)
        
        # Insert log record
        # Schema: pipeline_logs (id, name, state, created_at)
        # Use 'step' parameter as 'name' in the database
        query = f"""
            INSERT INTO {table_name} (name, state, created_at)
            VALUES (%s, %s, %s)
            RETURNING id
        """
        
        cursor.execute(query, (step, state_json, datetime.now()))
        log_id = cursor.fetchone()[0]
        conn.commit()
        
        logger.info(f"Logged state to database (id: {log_id}, name: {step})")
        return log_id
        
    except Exception as e:
        if conn:
            conn.rollback()
        logger.error(f"Failed to log state to database: {e}")
        return None
    finally:
        if conn:
            return_connection(conn)


def close_connection_pool():
    """
    Close all connections in the pool.
    """
    global _connection_pool
    if _connection_pool:
        _connection_pool.closeall()
        _connection_pool = None
        logger.info("PostgreSQL connection pool closed")

def log_feedback_to_db(
    pipeline_id: str,
    node_ids: List[str],
    user_decision: str,
    is_mapping: bool,
    user_feedback_text: str = None,
    bpmn: Dict[str, Any] = None,
    mapping: List[Dict] = None,
    node_mapping_feedback: List[str] = None
) -> Optional[int]:
    """
    Log user feedback to feedback_log table.
    
    Args:
        pipeline_id: thread_id of the pipeline_logs record (VARCHAR)
        node_ids: List of node IDs related to this feedback
        user_decision: User decision ("approve" or "reject")
        is_mapping: True if this is mapping feedback, False if BPMN feedback
        user_feedback_text: Feedback text from user
        bpmn: BPMN data (JSONB)
        mapping: Mapping data (JSONB)
        node_mapping_feedback: List of node IDs with mapping feedback
    
    Returns:
        feedback_id: ID of the inserted feedback record, or None if failed
    """
    if not PSYCOPG2_AVAILABLE:
        logger.warning("psycopg2 not available, skipping feedback log")
        return None
    
    conn = None
    try:
        conn = get_connection()
        cursor = conn.cursor()
        
        # Convert to JSONB
        bpmn_json = Json(bpmn) if bpmn else None
        mapping_json = Json(mapping) if mapping else None
        
        # Insert feedback log
        query = """
            INSERT INTO feedback_log (
                thread_id, node_ids, user_decision, is_mapping,
                node_mapping_feedback, bpmn, mapping, user_feedback
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            RETURNING id
        """
        
        cursor.execute(query, (
            pipeline_id,  # pipeline_id is actually thread_id (string)
            node_ids,
            user_decision,
            is_mapping,
            node_mapping_feedback,
            bpmn_json,
            mapping_json,
            user_feedback_text
        ))
        feedback_id = cursor.fetchone()[0]
        conn.commit()
        
        logger.info(f"Logged feedback to database (id: {feedback_id}, pipeline_id: {pipeline_id})")
        return feedback_id
        
    except Exception as e:
        if conn:
            conn.rollback()
        logger.error(f"Failed to log feedback to database: {e}")
        return None
    finally:
        if conn:
            return_connection(conn)

def init_schema(schema_sql_path: str):
    """
    Initialize database schema from SQL file.
    """
    conn = None
    try:
        conn = get_connection()
        cursor = conn.cursor()

        with open(schema_sql_path, "r", encoding="utf-8") as f:
            schema_sql = f.read()

        cursor.execute(schema_sql)
        conn.commit()
        logger.info("Database schema initialized successfully")

    except Exception as e:
        if conn:
            conn.rollback()
        logger.error(f"Failed to initialize schema: {e}")
        raise

    finally:
        if conn:
            return_connection(conn)

def create_pipeline_log(thread_id: str, original_text: str) -> Optional[int]:
    """
    Create a new pipeline log record.
    """
    conn = None
    try:
        conn = get_connection()
        cursor = conn.cursor()
        query = """
            INSERT INTO pipeline_logs (thread_id, original_text, created_at)
            VALUES (%s, %s, %s)
        """
        cursor.execute(query, (thread_id, original_text, datetime.now()))
        conn.commit()
        logger.info(f"Created pipeline log record (thread_id: {thread_id})")
    except Exception as e:
        if conn:
            conn.rollback()
        logger.error(f"Failed to create pipeline log record: {e}")
        return None


if __name__ == "__main__":
    init_schema("app/logs/schema.sql")
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM pipeline_logs")
    print(cursor.fetchall())
    conn.close()
    cursor.close()
    close_connection_pool()