-- SQL Queries for BPMN Generation Evaluation
-- These queries are used by bpmn_generation_metrics.py
-- ============================================================================
-- Query 1: Get all BPMN generation feedback records
-- ============================================================================
-- This query retrieves all feedback records for the BPMN generation stage
-- (where is_mapping = false), ordered by thread_id and timestamp
SELECT id,
    timestamp,
    thread_id,
    node_ids,
    user_decision,
    is_mapping,
    node_mapping_feedback,
    bpmn,
    mapping,
    user_feedback
FROM feedback_log
WHERE is_mapping = false
ORDER BY thread_id,
    timestamp;
-- ============================================================================
-- Query 2: Count threads by approval status (for verification)
-- ============================================================================
-- This query can be used to verify first approval counts
WITH first_feedbacks AS (
    SELECT DISTINCT ON (thread_id) thread_id,
        user_decision,
        timestamp
    FROM feedback_log
    WHERE is_mapping = false
    ORDER BY thread_id,
        timestamp
)
SELECT user_decision,
    COUNT(*) as count
FROM first_feedbacks
GROUP BY user_decision;
-- ============================================================================
-- Query 3: Count threads by final approval status (for verification)
-- ============================================================================
-- This query can be used to verify final approval counts
WITH last_feedbacks AS (
    SELECT DISTINCT ON (thread_id) thread_id,
        user_decision,
        timestamp
    FROM feedback_log
    WHERE is_mapping = false
    ORDER BY thread_id,
        timestamp DESC
)
SELECT user_decision,
    COUNT(*) as count
FROM last_feedbacks
GROUP BY user_decision;
-- ============================================================================
-- Query 4: Get feedback iteration counts per thread
-- ============================================================================
-- This query shows how many feedback iterations each thread had
SELECT thread_id,
    COUNT(*) as feedback_count,
    MIN(timestamp) as first_feedback_time,
    MAX(timestamp) as last_feedback_time
FROM feedback_log
WHERE is_mapping = false
GROUP BY thread_id
ORDER BY feedback_count DESC;
-- ============================================================================
-- Query 5: Get threads with multiple feedback iterations
-- ============================================================================
-- This query identifies threads that required multiple feedback rounds
SELECT thread_id,
    COUNT(*) as iterations,
    STRING_AGG(
        user_decision::text,
        ' -> '
        ORDER BY timestamp
    ) as decision_flow
FROM feedback_log
WHERE is_mapping = false
GROUP BY thread_id
HAVING COUNT(*) > 1
ORDER BY iterations DESC;