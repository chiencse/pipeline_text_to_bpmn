-- PostgreSQL schema initialization for pipeline logs
-- Edit this file to define your log table structure
-- Example schema (you can modify this):
CREATE TABLE IF NOT EXISTS pipeline_logs (
    thread_id VARCHAR(255) PRIMARY KEY,
    original_text VARCHAR(255),
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
CREATE INDEX IF NOT EXISTS idx_pipeline_logs_timestamp ON pipeline_logs(created_at);
CREATE INDEX IF NOT EXISTS idx_pipeline_logs_original_text ON pipeline_logs(original_text);
CREATE INDEX IF NOT EXISTS idx_feedback_log_thread_id ON feedback_log(thread_id);
CREATE INDEX IF NOT EXISTS idx_feedback_log_timestamp ON feedback_log(timestamp);