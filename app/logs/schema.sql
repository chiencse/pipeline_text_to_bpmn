-- PostgreSQL schema initialization for pipeline logs
-- Edit this file to define your log table structure
-- Example schema (you can modify this):
CREATE TABLE IF NOT EXISTS pipeline_logs (
    thread_id VARCHAR(255) PRIMARY KEY,
    original_text TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE IF NOT EXISTS feedback_log (
    id SERIAL PRIMARY KEY,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    thread_id VARCHAR(255),
    node_ids VARCHAR(255) [],
    user_decision VARCHAR(255),
    is_mapping BOOLEAN,
    node_mapping_feedback VARCHAR(255) [],
    bpmn JSONB,
    mapping JSONB,
    user_feedback TEXT,
    FOREIGN KEY (thread_id) REFERENCES pipeline_logs(thread_id)
);
CREATE TABLE IF NOT EXISTS retrieval_scores (
    id SERIAL PRIMARY KEY,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    thread_id VARCHAR(255),
    node_id VARCHAR(255) NOT NULL,
    node_name TEXT,
    avg_total_score FLOAT,
    avg_bm25_score FLOAT,
    avg_cosine_score FLOAT,
    num_candidates INTEGER,
    candidates JSONB,
    FOREIGN KEY (thread_id) REFERENCES pipeline_logs(thread_id) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS idx_pipeline_logs_timestamp ON pipeline_logs(created_at);
CREATE INDEX IF NOT EXISTS idx_pipeline_logs_original_text ON pipeline_logs(original_text);
CREATE INDEX IF NOT EXISTS idx_feedback_log_thread_id ON feedback_log(thread_id);
CREATE INDEX IF NOT EXISTS idx_feedback_log_timestamp ON feedback_log(timestamp);
CREATE INDEX IF NOT EXISTS idx_retrieval_scores_thread_id ON retrieval_scores(thread_id);
CREATE INDEX IF NOT EXISTS idx_retrieval_scores_node_id ON retrieval_scores(node_id);
CREATE INDEX IF NOT EXISTS idx_retrieval_scores_timestamp ON retrieval_scores(timestamp);