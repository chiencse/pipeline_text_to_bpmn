import os
from dotenv import load_dotenv
import google.generativeai as genai
from typing import List, Dict
from app.schemas import GraphOutput

# --- Cấu hình Gemini model ---
# Ensure GEMINI_API_KEY is set in your environment variables before running the app.
# For local development, you can create a .env file and use python-dotenv to load it:

load_dotenv()  # This loads environment variables from .env if present

genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# Chọn model hỗ trợ JSON output
MODEL_NER = "gemini-2.0-flash"
MODEL_BPMN = "gemini-2.5-pro"  

# ========== CALL NER ==========
def call_llm_ner(state: List[str]) -> List[Dict]:
    """
    Gọi Gemini để thực hiện NER (Named Entity Recognition)
    
    Use POS + dependency info to:
    - Prefer nouns (NN, NNP, NNS, NNPS) as "object", "role", "system", "resource", or "process" depending on context and modifiers.
    - Use `nsubj`, `dobj`, `acl`, `conj`, `prep` relations to identify actor → action → object patterns.
    Dependencies parser:
    {syntax_parser}
    """
    syntax_parser = state.get("syntax", {})
    chunks = state.get("chunks", [])
    full_text = chunks[0].get("text", "") if chunks else ""
    prompt = f"""
    You are an advanced information extraction model specialized in Business Process Management (BPMN) and Enterprise Resource Planning (ERP) domains.

    Your task is to extract **semantic entities** that describe elements of business processes and system interactions.

    From the following text chunks, extract and classify entities into one of these categories:
    - "Action" → verbs or phrases that describe a process step or user/system activity
    - "Object" → business objects or data items being manipulated (e.g., request, form, order)
    - "Role" → human or system actors performing actions (e.g., officer, employee, system)
    - "System" → specific applications or modules (e.g., SAP, Odoo, CRM)
    - "Event" → triggers or conditions initiating or following a process (e.g., on approval, when form is submitted)
    - "DataField" → specific attributes, input fields, or information pieces (e.g., user ID, status)
    - "Condition" → logical or business rules controlling the process (e.g., if amount > 5000)
    - "Resource" → documents, templates, or other assets involved (e.g., approval form, report)
    - "Process" → explicit process or subprocess names (e.g., member onboarding process)
    - "Output" → outcomes or system responses (e.g., notification sent, confirmation message)

    Return ONLY a **valid JSON list**, where each item has the format:
    {{
      "text": "...",
      "label": "...",
      "confidence": 0.0
      "start": 0,
    }}

    Do not include explanations or text outside of the JSON.
    Text chunks:
    {full_text}
    """
    
    model = genai.GenerativeModel(MODEL_NER)
    response = model.generate_content(prompt, generation_config={"response_mime_type": "application/json",  "temperature": 0.1})
    print("Token used in NER call:", response.usage_metadata)
    try:
        entities = response.text
        import json
        return json.loads(entities)
    except Exception:
        print("⚠️ Gemini returned non-JSON, raw output:", response.text)
        return []

# ========== CALL BPMN WITH MAPPING ==========
def call_llm_bpmn_with_mapping(text: str, entities: List[Dict], candidates: List[Dict], syntax_parser: List[Dict]) -> GraphOutput:
    """
    Gọi Gemini để sinh BPMN flow + mapping.
    """
    model = genai.GenerativeModel(MODEL_BPMN)
    

    prompt = f"""
    You are a process modeling assistant in the context RPA. Generate a BPMN process model and map candidate activities to BPMN nodes.
    Context & high-level rules:
    - Use BPMN to represent the process. Each software robot (bot) must be represented as a subprocess assignment (bot id) via node.subprocess or mapping.bot_id.
    - If a task is fully automatable by a bot, mark mapping.is_automatic = true and assign mapping.bot_id (e.g., "bot1"). If a task requires a human, mark is_automatic = false, manual_review = true, and either set node.type = "ManualTask" or assign a separate human subprocess (e.g., "human_approver").
    - Bot permissions and responsibilities are expressed via node.pool and node.lane. Pools represent system boundaries (e.g., "SAP", "EmailSystem"); lanes represent roles or bot identities (e.g., "ProcurementBot", "FinanceBot", "HumanApprover").
    - For tasks that are partly automated (human-in-the-loop), prefer "UserTask" or "Task" + mapping.manual_review = true.
    - Create subprocess ids like "bot1", "bot2", "human1" when splitting the process across multiple bots/actors. Each bot should correspond to a meaningful subprocess that groups its tasks.
    Rules:
    - Output ONLY a single valid JSON object that strictly follows the GraphOutput schema below.
    - Do NOT include explanations, comments, markdown, or extra fields.
    - Entities are provided as a list of objects with: text, label, confidence (optional), start (optional), end (optional).
    - Use ONLY these entity labels: Action, Object, Role, System, Event, DataField, Condition, Resource, Process, Output.
    - Candidate activities are provided as a list of objects with: activity_id, pkg, keyword, score, confidence (optional).
    - Use dependency parser info to infer relationships and task sequence.
    - You do NOT need to use all candidates. You may ignore candidate activities that don't fit the text context. 
    - Type BPMN supported: StartEvent, EndEvent, Task, ManualTask, UserTask, Gateway(Exclusive, Parallel).
    - Ensure flows connect nodes logically based on process sequence Follow rule design bpmn process in BPMN 2.0,  e.g., upload must occur before send or email.

    Required JSON output structure:
    {{
      "intent": {{"code": "string", "confidence": 0.0}},
      "relations": [
        {{"head": "officer", "tail": "select menu", "type": "agent_of"}} // Literal["agent_of", "exec_in", "object_of", "receiver_of", "sequence_next", "condition_of", "use_of", "purpose_of", "result_of"]
      ],
      "bpmn": {{
        "nodes": [
          {{"id": "n1", "type": "StartEvent", "name": "Start", "lane": "Officer", " subprocess": ""}},
          {{"id": "n2", "type": "Task", "name": "Select menu", "lane": "Officer", " subprocess": "bot1"}}
          {{"id": "n3", "type": "ExclusiveGateway", "name": "Exclusive Gateway"}}
        ],
        "flows": [
          {{"source": "n1", "target": "n2", "type": "SequenceFlow", "condition": ""}},
          {{"source": "n2", "target": "n3", "type": "SequenceFlow"}}
        ]
      }},
      "mapping": [
        {{
          "node_id": "n2",
          "activity_id": "connect_to_sap_system",
          "confidence": 0.92,
          "manual_review": false,
          "bot_id": "bot1",
          "candidates": [
            {{"activity_id": "connect_to_sap_system", "score": 0.8843, "confidence": 0.96, "pkg": "rpa-sap-mock", "keyword": "connect sap"}}
          ],
          "input_bindings": {{}},
          "outputs": []
        }}
      ]
    }}

    Input:
    Entities: {entities}
    Candidate Activities: {candidates}
    Dependency Parser Info: {syntax_parser}
    Text Input: {text}

    Now produce ONLY the JSON output.
    """



    response = model.generate_content(prompt, generation_config={"response_mime_type": "application/json"})
    print("Token used in BPMN with mapping call:", response.usage_metadata)
    import json
    try:
       
        data = json.loads(response.text)
        return GraphOutput(**data)
    except Exception:
        print("⚠️ Gemini returned invalid JSON:", response.text)
        raise

# ========== CALL BPMN FREE (nếu không cần mapping) ==========
def call_llm_bpmn_free(text: str, entities: List[Dict], relations: List[Dict]) -> GraphOutput:
    model = genai.GenerativeModel(MODEL_BPMN)
    prompt = f"""
    You are a process modeling assistant in the context RPA. Generate a BPMN process model as JSON structure.
    - rules:
      - Do NOT include explanations, comments, markdown, or extra fields.
      - Type BPMN supported: StartEvent, EndEvent, Task, ManualTask, UserTask, Gateway(Exclusive, Parallel).
      - Ensure flows connect nodes logically based on process sequence Follow rule design bpmn process in BPMN 2.0  e.g., upload must occur before send or email, gateways split/join flows correctly.
      - If a task is fully automatable by a bot, mark mapping.is_automatic = true and assign mapping.bot_id (e.g., "bot1"). If a task requires a human, mark is_automatic = false, manual_review = true, and either set node.type = "ManualTask" or assign a separate human subprocess (e.g., "human_approver").

    Given the text user input context: {text}
    and entities: {entities}
    and relations: {relations}
    Read and analyze the above context to
    Generate BPMN process model as JSON structure:
      {{
      "intent": {{"code": "string", "confidence": 0.0}},
      "relations": [
        {{"head": "officer", "tail": "select menu", "type": "agent_of"}} // Literal["agent_of", "exec_in", "object_of", "receiver_of", "sequence_next"]
      ],
      "bpmn": {{
        "nodes": [
          {{"id": "n1", "type": "StartEvent", "name": "Start", "lane": "Officer", " subprocess": ""}},
          {{"id": "n2", "type": "Task", "name": "Select menu", "lane": "Officer", " subprocess": "bot1"}}
          {{"id": "n3", "type": "ExclusiveGateway", "name": "Exclusive Gateway"}}
        ],
        "flows": [
          {{"source": "n1", "target": "n2", "type": "SequenceFlow", "condition": ""}},
          {{"source": "n2", "target": "n3", "type": "SequenceFlow"}}
        ]
      }},
    }}

    """
    response = model.generate_content(prompt, generation_config={"response_mime_type": "application/json"})
    import json
    try:
        data = json.loads(response.text)
        return GraphOutput(**data)
    except Exception:
        print("⚠️ Gemini returned invalid JSON:", response.text)
        raise
