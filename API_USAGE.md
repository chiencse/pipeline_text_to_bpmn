# FastAPI Endpoints for Scenario B Pipeline

This document describes how to use the FastAPI endpoints for Scenario B pipeline with user feedback support.

## Overview

The pipeline now supports two user feedback checkpoints:
1. **BPMN Feedback**: After generating the BPMN structure, user can approve or reject
2. **Mapping Feedback**: After mapping activities to BPMN nodes, user can approve or reject

## Endpoints

### 1. Start Pipeline
**POST** `/pipeline/b/start`

Start a new pipeline execution with a unique thread_id.

**Request Body:**
```json
{
  "text": "Send an email to finance, attach the quotation, wait for reply, then update the invoice in the SAP system.",
  "options": {}
}
```

**Response:**
```json
{
  "thread_id": "uuid-string",
  "status": "waiting_feedback",
  "interrupt": {
    "type": "bpmn_feedback",
    "instruction": "Review the generated BPMN structure. Do you approve it?",
    "bpmn": { ... },
    "draft_snapshot": { ... }
  },
  "state": { ... }
}
```

### 2. Get Pending Feedback
**GET** `/pipeline/b/feedback/{thread_id}`

Check if there's a pending feedback request for a thread.

**Response:**
```json
{
  "status": "waiting_feedback",
  "interrupt": {
    "type": "bpmn_feedback",
    "instruction": "...",
    "bpmn": { ... }
  },
  "node": "user_feedback",
  "state": { ... }
}
```

### 3. Submit Feedback
**POST** `/pipeline/b/feedback/{thread_id}`

Submit user feedback to resume the pipeline.

**Request Body (for BPMN feedback):**
```json
{
  "user_decision": "approve"  // or "reject"
}
```

**Request Body (for BPMN feedback - reject with feedback):**
```json
{
  "user_decision": "reject",
  "user_feedback_text": "Please add error handling for email sending"
}
```

**Request Body (for Mapping feedback - approve):**
```json
{
  "user_mapping_decision": "approve"
}
```

**Request Body (for Mapping feedback - reject):**
```json
{
  "user_mapping_decision": "reject",
  "user_mapping_feedback_text": "The activity mapping for node X is incorrect"
}
```

**Response:**
```json
{
  "thread_id": "uuid-string",
  "status": "waiting_feedback",  // or "completed"
  "interrupt": { ... },  // if another feedback is needed
  "state": { ... }
}
```

### 4. Get Pipeline Status
**GET** `/pipeline/b/status/{thread_id}`

Get the current status of a pipeline thread.

**Response:**
```json
{
  "status": "waiting_feedback" | "running" | "completed",
  "interrupt": { ... },  // if waiting_feedback
  "current_node": "node_name",  // if running
  "state": { ... }
}
```

## Workflow Example

1. **Start Pipeline:**
   ```bash
   POST /pipeline/b/start
   {
     "text": "Send email and create spreadsheet"
   }
   ```
   Returns `thread_id` and first interrupt (BPMN feedback)

2. **Check Feedback (optional polling):**
   ```bash
   GET /pipeline/b/feedback/{thread_id}
   ```

3. **Submit BPMN Feedback:**
   ```bash
   POST /pipeline/b/feedback/{thread_id}
   {
     "user_decision": "approve"
   }
   ```
   Pipeline continues to retrieve_map, then pauses for mapping feedback

4. **Submit Mapping Feedback:**
   ```bash
   POST /pipeline/b/feedback/{thread_id}
   {
     "user_mapping_decision": "approve"
   }
   ```
   Pipeline continues to validate and render

5. **Check Final Status:**
   ```bash
   GET /pipeline/b/status/{thread_id}
   ```
   Returns completed state with render_xml, render_activities, etc.

## Feedback Types

### BPMN Feedback (`user_feedback` node)
- **Approve**: Continue to `retrieve_map`
- **Reject**: Go back to `bpmn_free` with feedback text incorporated

### Mapping Feedback (`user_feedback_mapping` node)
- **Approve**: Continue to `validate`
- **Reject**: Go back to `retrieve_map` with feedback text incorporated

## Error Handling

- If thread_id not found: 404 error
- If no pending feedback: Returns current status
- If pipeline error: 500 error with details

## CORS

CORS middleware is enabled for all origins. Configure `allow_origins` in production.



