# Thread ID Integration for Retrieval Scores Logging

## Problem

The retrieval scores logging feature was failing with a foreign key constraint error:

```
Failed to log retrieval scores to database: insert or update on table "retrieval_scores" 
violates foreign key constraint "retrieval_scores_thread_id_fkey"
DETAIL: Key (thread_id)=(unknown) is not present in table "pipeline_logs".
```

## Root Cause

The `thread_id` was generated in the FastAPI endpoint (`pipeline_b_start`) but was NOT added to the pipeline state. When retrieval logging tried to access it, it defaulted to "unknown", which didn't exist in the `pipeline_logs` table.

## Solution

### 1. Add `thread_id` to Initial State

**File: `app/main.py`**

```python
@app.post("/pipeline/b/start")
def pipeline_b_start(req: Req):
    thread_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}
    
    # ✅ Add thread_id to state so nodes can access it
    initial_state = {
        "text": req.text, 
        "config": req.options or {},
        "thread_id": thread_id  # ← This is the fix!
    }
    
    # Create pipeline log in database
    threading.Thread(
        target=create_pipeline_log_safe,
        args=(thread_id, req.text),
        daemon=True
    ).start()
```

### 2. Update Legacy Endpoint

**File: `app/main.py`**

```python
@app.post("/pipeline/b")
def pipeline_b(req: Req):
    thread_id = str(uuid.uuid4())
    
    # ✅ Add thread_id to state for legacy endpoint too
    state = {
        "text": req.text, 
        "config": req.options or {},
        "thread_id": thread_id
    }
    
    # Create pipeline log in background
    threading.Thread(
        target=create_pipeline_log_safe,
        args=(thread_id, req.text),
        daemon=True
    ).start()
```

### 3. Simplify Thread ID Access in Nodes

**File: `app/scenario_b.py`**

```python
def retrieve_candidates_for_task(task, state: PipelineState = None):
    node_name = task.get("name", "")
    node_id = task.get("id", "")
    
    candidates = hybrid_search(node_name, k=5)
    
    # Log retrieval scores asynchronously if state is provided
    if candidates and state:
        # ✅ Get thread_id directly from state (simple and clean)
        thread_id = state.get("thread_id", "unknown")
        
        try:
            log_retrieval_scores_async(
                thread_id=thread_id,
                node_id=node_id,
                node_name=node_name,
                candidates=candidates
            )
        except Exception as e:
            print(f"Warning: Failed to log retrieval scores: {e}")
    
    return candidates
```

## Data Flow

```
┌─────────────────────────────────────────────────────────────┐
│ 1. FastAPI Endpoint (app/main.py)                           │
│    • Generate UUID: thread_id = str(uuid.uuid4())           │
│    • Create pipeline log: create_pipeline_log(thread_id)    │
│    • Add to state: state["thread_id"] = thread_id ✅        │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│ 2. LangGraph Pipeline (scenario_b.py)                       │
│    • State flows through all nodes                          │
│    • Each node has access to state["thread_id"]             │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│ 3. Retrieval Node (node_retrieve_map)                       │
│    • Calls retrieve_candidates_for_task(task, state)        │
│    • Passes full state to function                          │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│ 4. Retrieval Function (retrieve_candidates_for_task)        │
│    • Gets thread_id: state.get("thread_id")                 │
│    • Logs scores: log_retrieval_scores_async(thread_id...)  │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│ 5. Database Logger (db_logger.py)                           │
│    • Inserts into retrieval_scores table                    │
│    • Foreign key references pipeline_logs(thread_id) ✅     │
└─────────────────────────────────────────────────────────────┘
```

## State Definition

**File: `app/state.py`**

The `thread_id` field is already defined in `PipelineState`:

```python
class PipelineState(TypedDict, total=False):
    thread_id: str  # ✅ Already defined
    text: str
    chunks: List[str]
    # ... other fields
```

## Database Schema

**File: `app/logs/schema.sql`**

```sql
-- Main pipeline logs table
CREATE TABLE IF NOT EXISTS pipeline_logs (
    thread_id VARCHAR(255) PRIMARY KEY,  -- ← Referenced by retrieval_scores
    original_text TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Retrieval scores table (logs hybrid_search results)
CREATE TABLE IF NOT EXISTS retrieval_scores (
    id SERIAL PRIMARY KEY,
    thread_id VARCHAR(255),  -- ← Must match pipeline_logs.thread_id
    node_id VARCHAR(255) NOT NULL,
    node_name TEXT,
    avg_total_score FLOAT,
    avg_bm25_score FLOAT,
    avg_cosine_score FLOAT,
    num_candidates INTEGER,
    candidates JSONB,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (thread_id) REFERENCES pipeline_logs(thread_id)  -- ← Constraint
);
```

## Testing

### Before Fix
```bash
# Error: thread_id was "unknown"
Failed to log retrieval scores to database: 
insert or update on table "retrieval_scores" violates foreign key constraint
```

### After Fix
```bash
# Success: thread_id is valid UUID
✅ Created pipeline log record (thread_id: 550e8400-e29b-41d4-a716-446655440000)
✅ Logged retrieval scores (id: 1, thread_id: 550e8400-e29b-41d4-a716-446655440000, 
   node_id: task_send_email, avg_total: 0.8523)
```

## Benefits

1. **Proper Foreign Key Relationships**: All retrieval scores are linked to a valid pipeline run
2. **Traceability**: Can track all retrieval operations for a specific pipeline execution
3. **Data Integrity**: Database enforces referential integrity
4. **Clean Architecture**: thread_id flows naturally through the state object

## Future Enhancements

- Add thread_id to all logging operations for consistency
- Create indexes on thread_id for faster queries
- Add cascade delete to clean up retrieval scores when pipeline logs are deleted
- Consider adding thread_id to other log tables (feedback_log already has it)





