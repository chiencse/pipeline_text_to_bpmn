import os
import json
from dotenv import load_dotenv
from typing import List, Dict, Any
from langchain_google_genai import ChatGoogleGenerativeAI
from app.schemas import GraphOutput

# --- Cấu hình Gemini model ---
# Ensure GEMINI_API_KEY is set in your environment variables before running the app.
# For local development, you can create a .env file and use python-dotenv to load it:

load_dotenv()  # This loads environment variables from .env if present

# Chọn model hỗ trợ JSON output
MODEL_NER = "gemini-2.5-flash"
MODEL_BPMN = "gemini-2.5-flash"

# Initialize LangChain Google Generative AI models
def get_ner_model():
    """Get NER model instance."""
    return ChatGoogleGenerativeAI(
        model=MODEL_NER,
        google_api_key=os.getenv("GEMINI_API_KEY"),
        temperature=0.1,
        convert_system_message_to_human=True,
    )

def get_bpmn_model():
    """Get BPMN model instance."""
    return ChatGoogleGenerativeAI(
        model=MODEL_BPMN,
        google_api_key=os.getenv("GEMINI_API_KEY"),
        temperature=0.1,
        convert_system_message_to_human=True,
    )  

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
    
    model = get_ner_model()
    
    # LangChain invoke - prompt đã có instruction về JSON format
    try:
        # Thêm instruction rõ ràng về JSON format vào đầu prompt
        json_prompt = "You must respond with valid JSON only. " + prompt
        
        response = model.invoke(json_prompt)
        
        # LangChain trả về AIMessage, lấy content
        response_text = response.content if hasattr(response, 'content') else str(response)
        
        # Log usage metadata nếu có
        if hasattr(response, 'response_metadata'):
            print("Token used in NER call:", response.response_metadata)
        
        # Clean response text - loại bỏ markdown code blocks nếu có
        response_text = response_text.strip()
        if response_text.startswith("```json"):
            response_text = response_text[7:]
        if response_text.startswith("```"):
            response_text = response_text[3:]
        if response_text.endswith("```"):
            response_text = response_text[:-3]
        response_text = response_text.strip()
        
        # Parse JSON
        entities = json.loads(response_text)
        
        # Đảm bảo trả về list
        if isinstance(entities, dict) and "entities" in entities:
            entities = entities["entities"]
        elif not isinstance(entities, list):
            entities = [entities] if entities else []
            
        return entities
    except json.JSONDecodeError as e:
        print(f"⚠️ Gemini returned non-JSON, raw output: {response_text}")
        print(f"JSON decode error: {e}")
        return []
    except Exception as e:
        print(f"⚠️ Error in NER call: {e}")
        return []

# ========== CALL BPMN WITH MAPPING ==========
def call_llm_bpmn_with_mapping(text: str, entities: List[Dict], candidates: List[Dict], syntax_parser: List[Dict]) -> GraphOutput:
    """
    Gọi Gemini để sinh BPMN flow + mapping.
    """
    model = get_bpmn_model()
    
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



    try:
        # Thêm instruction rõ ràng về JSON format vào đầu prompt
        json_prompt = "You must respond with valid JSON only. " + prompt
        
        response = model.invoke(json_prompt)
        
        # LangChain trả về AIMessage, lấy content
        response_text = response.content if hasattr(response, 'content') else str(response)
        
        # Log usage metadata nếu có
        if hasattr(response, 'response_metadata'):
            print("Token used in BPMN with mapping call:", response.response_metadata)
        
        # Clean response text - loại bỏ markdown code blocks nếu có
        response_text = response_text.strip()
        if response_text.startswith("```json"):
            response_text = response_text[7:]
        if response_text.startswith("```"):
            response_text = response_text[3:]
        if response_text.endswith("```"):
            response_text = response_text[:-3]
        response_text = response_text.strip()
        
        # Parse JSON và tạo GraphOutput
        data = json.loads(response_text)
        return GraphOutput(**data)
    except json.JSONDecodeError as e:
        print(f"⚠️ Gemini returned invalid JSON: {response_text}")
        print(f"JSON decode error: {e}")
        raise
    except Exception as e:
        print(f"⚠️ Error in BPMN with mapping call: {e}")
        raise

# ========== CALL BPMN FREE (nếu không cần mapping) ==========
def call_llm_bpmn_free(text: str, entities: List[Dict], relations: List[Dict]) -> GraphOutput:
    model = get_bpmn_model()
    prompt = f"""
    You are a process modeling assistant in the context RPA. Your goal is to convert natural language process descriptions into a strict JSON structure representing a BPMN diagram.
   ### RULES & CONSTRAINTS:
    1. **Output Format**: Return ONLY valid JSON. Do not include markdown formatting (```json), comments, or explanations. 
       - "queryActivity" is the string query(name/keyword + description + contextPackage) should use to get the retrieval similar activities(Education Moodle, Email, Google Sheet, Google Drive, Control, ...)
    2. **Lane/Actor Identification**: Identify the actor performing each step. Assign them to the "lane" field.
    3. **Node Type Selection Guidelines**: - Inspect all cases and select the most appropriate node type,  can base on keyword of user input.
        - **StartEvent/EndEvent**: Marking the beginning and completion of the flow. Only one start/end event is allowed.
        - **ManualTask**: A human performs a physical task without software (e.g., "Sign paper", "Print document","grade, count number of sheets").
        - **UserTask**: A human performs a task using software/computer, human interaction tasks ( reviewing, deciding, entering data, e.g., "Check email", "Enter data").
        - **ServiceTask** (or generic "Task" with is_automatic=true): A fully automated step performed by a system or bot.
        - **ExclusiveGateway**: Used for "If/Else" decisions or loops (checking conditions).
        - **ParallelGateway**: Used when tasks happen simultaneously.
        - **ReceiveTask/SendTask**: for automated receiving operations/ automated sending operations (e.g., "Receive email", "Send email").
        - **Task**: A task is a step in the process that is performed by a system or bot. only if the action cannot be clearly classified as any of the above types.
    4. **RPA Automation Logic**:
        - If a task involves calculation, data entry, or digital verification that implies NO human intervention, set `"is_automatic": true` and assign a `"bot_id"` (e.g., "bot_1").
        - If a task is explicitly human (review, approve, physical action), set `"is_automatic": false`.

    5. **Flow Logic**:
        - Ensure every split (Gateway) eventually merges or leads to end events.
        - Handle loops by connecting a SequenceFlow back to a previous Gateway or Task. if the task is in a loop, set "in_loop": true.

    Given the text user input context: {text}

 
    Read and analyze the above context to
    Generate BPMN process model as JSON structure:
      {{
      "intent": {{"code": "string", "confidence": 0.0}},
      "bpmn": {{
        "nodes": [
          {{"queryActivity": "send email to officer",
            "id": "n1", "type": "StartEvent", "name": "Start", "lane": "Officer", "in_loop": boolean, "automation": {{
            "is_automatic": boolean, 
            "bot_id": "string or null",
            "manual_review_required": boolean
        }}}},
          {{
            "queryActivity": "create google sheet",
            "id": "n2", "type": "Task", "name": "Select menu", "lane": "Officer", "in_loop": boolean, "automation": {{
            "is_automatic": boolean, 
            "bot_id": "bot1",
            "manual_review_required": boolean
        }}}},
          {{"queryActivity": "if/else loop condition for count", "id": "n3", "type": "ExclusiveGateway", "name": "Exclusive Gateway", "in_loop": boolean}}
        ],
        "flows": [
          {{"source": "n1", "target": "n2", "type": "SequenceFlow", "condition": "string (only for flows coming out of ExclusiveGateway)"}},
          {{"source": "n2", "target": "n3", "type": "SequenceFlow"}}
        ]
      }},
    }}

    """
    try:
        # Thêm instruction rõ ràng về JSON format vào đầu prompt
        json_prompt = prompt
        
        response = model.invoke(json_prompt)
        
        # LangChain trả về AIMessage, lấy content
        response_text = response.content if hasattr(response, 'content') else str(response)
        
        # Log usage metadata nếu có
        if hasattr(response, 'response_metadata'):
            print("Token used in BPMN free call:", response.response_metadata)
        
        # Clean response text - loại bỏ markdown code blocks nếu có
        response_text = response_text.strip()
        if response_text.startswith("```json"):
            response_text = response_text[7:]
        if response_text.startswith("```"):
            response_text = response_text[3:]
        if response_text.endswith("```"):
            response_text = response_text[:-3]
        response_text = response_text.strip()
        
        # Parse JSON và tạo GraphOutput
        data = json.loads(response_text)
        return GraphOutput(**data)
    except json.JSONDecodeError as e:
        print(f"⚠️ Gemini returned invalid JSON: {response_text}")
        print(f"JSON decode error: {e}")
        raise
    except Exception as e:
        print(f"⚠️ Error in BPMN free call: {e}")
        raise

# ========== CALL BPMN WITH FEEDBACK (chỉnh sửa BPMN theo feedback) ==========
def call_llm_bpmn_with_feedback(
    original_text: str,
    current_bpmn: Dict[str, Any],
    user_feedback_text: str,
    selected_node_ids: List[str] = None
) -> GraphOutput:
    """
    Gọi Gemini để chỉnh sửa BPMN flow dựa trên user feedback.
    Giữ nguyên các rule ban đầu nhưng đổi mục đích là chỉnh sửa theo feedback.
    """
    model = get_bpmn_model()
    
    prompt = f"""
    You are a process modeling assistant in the context RPA. Your task is to REVISE an existing BPMN process model based on user feedback.
    
    **IMPORTANT**: You must follow ALL the same rules and constraints as when creating a new BPMN model, but now your goal is to MODIFY the existing model according to the user's feedback.
    
    **Context & high-level rules (same as initial creation):**
    - Use BPMN to represent the process. Each software robot (bot) must be represented as a subprocess assignment (bot id) via node.subprocess or mapping.bot_id.
    - If a task is fully automatable by a bot, mark mapping.is_automatic = true and assign mapping.bot_id (e.g., "bot1"). If a task requires a human, mark is_automatic = false, manual_review = true, and either set node.type = "ManualTask" or assign a separate human subprocess (e.g., "human_approver").
    - Bot permissions and responsibilities are expressed via node.pool and node.lane. Pools represent system boundaries (e.g., "SAP", "EmailSystem"); lanes represent roles or bot identities (e.g., "ProcurementBot", "FinanceBot", "HumanApprover").
    - For tasks that are partly automated (human-in-the-loop), prefer "UserTask" or "Task" + mapping.manual_review = true.
    - Create subprocess ids like "bot1", "bot2", "human1" when splitting the process across multiple bots/actors. Each bot should correspond to a meaningful subprocess that groups its tasks.
    
    **Rules:**
    - Output ONLY a single valid JSON object that strictly follows the GraphOutput schema below.
    - Do NOT include explanations, comments, markdown, or extra fields.
    - Entities are provided as a list of objects with: text, label, confidence (optional), start (optional), end (optional).
    - Use ONLY these entity labels: Action, Object, Role, System, Event, DataField, Condition, Resource, Process, Output.
    - Use dependency parser info to infer relationships and task sequence.
    - Type BPMN supported: StartEvent, EndEvent, Task, ManualTask, UserTask, Gateway(Exclusive, Parallel).
    - Ensure flows connect nodes logically based on process sequence. Follow rule design bpmn process in BPMN 2.0, e.g., upload must occur before send or email.
    
    **Your task:**
    - Review the CURRENT BPMN model provided below
    - Understand the user's feedback
    - Revise the BPMN model to address the feedback while maintaining process logic
    - Keep valid parts of the current model that are not mentioned in feedback
    - Apply changes requested in the feedback
    
    **Current BPMN Model:**
    {json.dumps(current_bpmn, indent=2, ensure_ascii=False)}
    
    **User Feedback:**
    {user_feedback_text}
    
    **Selected Node IDs (nodes to revise):**
    {selected_node_ids if selected_node_ids else "All nodes"}
    
    **Original Input:**
    Text: {original_text}
    
    **Required JSON output structure:**
    {{
      "bpmn": {{
        "nodes": [
          {{"id": "n1", "type": "StartEvent", "name": "Start", "lane": "Officer", "subprocess": ""}},
          {{"id": "n2", "type": "Task", "name": "Select menu", "lane": "Officer", "subprocess": "bot1"}},
          {{"id": "n3", "type": "ExclusiveGateway", "name": "Exclusive Gateway"}}
        ],
        "flows": [
          {{"source": "n1", "target": "n2", "type": "SequenceFlow", "condition": ""}},
          {{"source": "n2", "target": "n3", "type": "SequenceFlow"}}
        ]
      }}
    }}
    
    Now produce ONLY the revised JSON output based on the feedback.
    """
    
    try:
        json_prompt = "You must respond with valid JSON only. " + prompt
        
        response = model.invoke(json_prompt)
        response_text = response.content if hasattr(response, 'content') else str(response)
        
        if hasattr(response, 'response_metadata'):
            print("Token used in BPMN with feedback call:", response.response_metadata)
        
        # Clean response text
        response_text = response_text.strip()
        if response_text.startswith("```json"):
            response_text = response_text[7:]
        if response_text.startswith("```"):
            response_text = response_text[3:]
        if response_text.endswith("```"):
            response_text = response_text[:-3]
        response_text = response_text.strip()
        
        # Parse JSON và tạo GraphOutput
        data = json.loads(response_text)
        return GraphOutput(**data)
    except json.JSONDecodeError as e:
        print(f"⚠️ Gemini returned invalid JSON for BPMN feedback: {response_text}")
        print(f"JSON decode error: {e}")
        raise
    except Exception as e:
        print(f"⚠️ Error in BPMN with feedback call: {e}")
        raise

# ========== CALL MAPPING WITH FEEDBACK (chỉnh sửa mapping theo feedback) ==========
def call_llm_mapping_with_feedback(
    original_text: str,
    current_bpmn: Dict[str, Any],
    current_mapping: List[Dict],
    candidates: List[Dict],
    user_feedback_text: str,
    selected_node_ids: List[str] = None
) -> List[Dict]:
    """
    Gọi Gemini để chỉnh sửa mapping dựa trên user feedback.
    Giữ nguyên các rule ban đầu nhưng đổi mục đích là chỉnh sửa theo feedback.
    
    Args:
        original_text: Text gốc từ user
        entities: Danh sách entities
        current_bpmn: BPMN hiện tại
        current_mapping: Mapping hiện tại cần chỉnh sửa
        candidates: Danh sách activity candidates
        user_feedback_text: Feedback text từ user
        selected_node_ids: Danh sách node IDs được chọn để chỉnh sửa (optional)
    
    Returns:
        List[Dict]: Updated mapping list
    """
    model = get_bpmn_model()
    
    # Filter candidates for selected nodes if provided
    if selected_node_ids:
        # Get candidates for selected nodes only
        filtered_candidates = {}
        for node_id in selected_node_ids:
            # Find candidates for this node (you may need to adjust this based on your data structure)
            node_candidates = [c for c in candidates if c.get("node_id") == node_id]
            if node_candidates:
                filtered_candidates[node_id] = node_candidates[:5]  # Top 5
    else:
        # Use all candidates
        filtered_candidates = candidates
    
    prompt = f"""
    You are an RPA (Robotic Process Automation) system expert. Your task is to REVISE activity candidate mappings for BPMN nodes based on user feedback.
    
    **IMPORTANT**: You must follow ALL the same rules and evaluation criteria as when creating initial mappings, but now your goal is to MODIFY the existing mappings according to the user's feedback.
    
    **Context & Rules (same as initial mapping):**
    - Use BPMN to represent the process. Each software robot (bot) must be represented as a subprocess assignment (bot id) via node.subprocess or mapping.bot_id.
    - If a task is fully automatable by a bot, mark mapping.is_automatic = true and assign mapping.bot_id (e.g., "bot1"). If a task requires a human, mark is_automatic = false, manual_review = true.
    - Bot permissions and responsibilities are expressed via node.pool and node.lane.
    - For tasks that are partly automated (human-in-the-loop), prefer "UserTask" or "Task" + mapping.manual_review = true.
    
    **Evaluation Criteria (same as initial):**
    1. **Automation Feasibility**: Can this task be fully automated by an RPA bot?
    2. **Activity Matching**: Select the most suitable activity candidate based on similarity and context.
    3. **Bot Assignment**: Assign appropriate bot_id for automated tasks.
    
    **Your task:**
    - Review the CURRENT mapping provided below
    - Understand the user's feedback
    - Revise the mappings to address the feedback
    - Keep valid mappings that are not mentioned in feedback
    - Apply changes requested in the feedback for selected nodes (if specified)
    
    **Current BPMN:**
    {json.dumps(current_bpmn, indent=2, ensure_ascii=False)}
    
    **Current Mapping:**
    {json.dumps(current_mapping, indent=2, ensure_ascii=False)}
    
    **Available Activity Candidates:**
    {json.dumps(filtered_candidates, indent=2, ensure_ascii=False)}
    
    **User Feedback:**
    {user_feedback_text}
    
    **Selected Node IDs (nodes to revise):**
    {selected_node_ids if selected_node_ids else "All nodes"}
    
    **Original Input:**
    Text: {original_text}
    
**Output Format:**
Return ONLY a valid JSON array. Each object should have:
{{
  "node_id": "string (required)",
  "is_automatic": boolean,
  "confidence": float (0.0-1.0),
  "selected_activity_id": "string or null",
  "reasoning": "string (brief explanation)",
}}
Example output:
[
  {{
    "node_id": "n1",
    "is_automatic": true,
    "confidence": 0.85,
    "selected_activity_id": "send_email_gmail",
    "reasoning": "Strong match with Gmail send activity. Task involves sending email which is fully automatable.",
  }},
  {{
    "node_id": "n2",
    "is_automatic": false,
    "confidence": 0.2,
    "selected_activity_id": null,
    "reasoning": "Task requires human judgment and approval. No suitable automation candidates.",
  }}
]

    
    Include ALL nodes from current mapping, but update only those mentioned in feedback or in selected_node_ids.
    
    Now produce ONLY the revised JSON array.
    """
    
    try:
        json_prompt = "You must respond with valid JSON only. " + prompt
        
        response = model.invoke(json_prompt)
        response_text = response.content if hasattr(response, 'content') else str(response)
        
        if hasattr(response, 'response_metadata'):
            print("Token used in mapping with feedback call:", response.response_metadata)
        
        # Clean response text
        response_text = response_text.strip()
        if response_text.startswith("```json"):
            response_text = response_text[7:]
        if response_text.startswith("```"):
            response_text = response_text[3:]
        if response_text.endswith("```"):
            response_text = response_text[:-3]
        response_text = response_text.strip()
        
        # Parse JSON
        updated_mapping = json.loads(response_text)
        
        # Ensure it's a list
        if isinstance(updated_mapping, dict):
            updated_mapping = [updated_mapping]
        elif not isinstance(updated_mapping, list):
            updated_mapping = []
        
        return updated_mapping
        
    except json.JSONDecodeError as e:
        print(f"⚠️ LLM returned invalid JSON for mapping feedback: {response_text}")
        print(f"JSON decode error: {e}")
        return current_mapping  # Return original if parsing fails
    except Exception as e:
        print(f"⚠️ Error in mapping with feedback call: {e}")
        return current_mapping  # Return original if error

# ========== CALL LLM EVALUATE AUTOMATION FEASIBILITY ==========
def call_llm_evaluate_automation_feasibility(
    nodes: List[Dict],
    flows: List[Dict],
    act_candidates: Dict[str, List[Dict]],
    original_text: str = ""
) -> List[Dict]:
    """
    Gọi LLM để đánh giá tính khả thi tự động hóa của các node trong BPMN.
    
    Đầu vào:
    - nodes: Danh sách các node trong BPMN
    - flows: Danh sách các flow kết nối nodes
    - act_candidates: Dict với key là node_id, value là list top 5 activity candidates (có score)
    - original_text: Text gốc từ user input (optional, để cung cấp context)
    
    Trả về:
    - List[Dict]: Mỗi dict chứa:
      {
        "node_id": str,
        "is_automatic": bool,
        "confidence": float (0.0-1.0),
        "selected_activity_id": str | None,
        "reasoning": str,
        "requires_manual_review": bool
      }
    """
    model = get_bpmn_model()
    
    # Format nodes info
    nodes_info = []
    for node in nodes:
        node_id = node.get("id", "")
        node_name = node.get("name", "")
        node_type = node.get("type", "")
        node_lane = node.get("lane", "")
        candidates = act_candidates.get(node_id, [])
        
        nodes_info.append({
            "id": node_id,
            "name": node_name,
            "in_loop": node.get("in_loop", False),
            "type": node_type,
            "lane": node_lane,
            "candidates": [
                {
                    "activity_id": c.get("activity_id", ""),
                    "pkg": c.get("pkg", ""),
                    "keyword": c.get("keyword", ""),
                    "score": c.get("score", 0.0),
                    "text": c.get("text", "")[:200]  # Limit text length
                }
                for c in candidates[:5]  # Top 5 only
            ]
        })
    
    # Format flows info for context
    flows_summary = []
    for flow in flows:
        source = flow.get("source", "")
        target = flow.get("target", "")
        condition = flow.get("condition", "")
        flows_summary.append({
            "from": source,
            "to": target,
            "condition": condition
        })
    
    prompt = f"""You are an RPA (Robotic Process Automation) system expert. Your task is to evaluate which BPMN nodes can be feasibly automated using the available RPA activity templates.

**Context:**
Original user intent: {original_text[:500] if original_text else "Not provided"}

**Available RPA Activity Candidates:**
For each node, System have been provided with top 5 activity candidates (retrieved via semantic search) that the RPA system supports. Each candidate has:
- activity_id: Unique identifier
- pkg: Package name (e.g., "gmail", "google_sheets", "rpa-sap-mock", "control", "data_manipulation")
- keyword: Short description
- score: Similarity score (0.0-1.0, higher is better)
- text: Description of what the activity does

**Evaluation Criteria:**
Base on user input context and available candidates, we will evaluate the automation feasibility of the task.
Considering all candidates, even those with high scores may not be suitable for the task in this context.
1. **Automation Feasibility**: Can this task be fully automated by an RPA bot?
 A task is considered AUTOMATABLE if:
- If a node has "in_loop": true, automation is strongly preferred.
- If a node has activiy can call/execute on the system, automation is strongly preferred.
- It belongs to common Education RPA scenarios:
  - Creating, uploading, updating, deleting, or actions to Education entities like courses, classes, students, teachers, etc.
  - Grading or calculating final marks (grading does NOT require a document template)
- A suitable activity candidate always exists OR the task can be handled by:
  - Gateway nodes with Control flow (if, for, while, loop, count, compare) - Use default suitable activity template "control" for control flow.
  - "data_manipulation" package for calculations or transformations
- If a node has "in_loop": true, automation is strongly preferred.

A task is considered NOT AUTOMATABLE if:
- It requires human judgment, subjective interpretation, creativity, or asking questions.
- It involves physical-world actions.
- not potentially automatable or no suitable activity exists and the task meaning implies human involvement.
  (except generic education control or data manipulation tasks).

**Important Rules:**
- Do NOT try to map every node. Select at most ONE suitable activity template candidate(with education grading assign "OCR")
- Gateway nodes with Question cannot calculate to compare so we cannot automate this task.
- ManualTask and UserTask types generally indicate human involvement - evaluate carefully.



**Input Data:**
Nodes to evaluate:
{json.dumps(nodes_info, indent=2, ensure_ascii=False)}

Flow connections (for context):
{json.dumps(flows_summary, indent=2, ensure_ascii=False)}

**Output Format:**
Return ONLY a valid JSON array. Each object should have:
{{
  "node_id": "string (required)",
  "is_automatic": boolean,
  "confidence": float (0.0-1.0),
  "selected_activity_id": "string or null",
  "reasoning": "string (brief explanation)",
}}

Only include nodes that you evaluated. Skip nodes that are clearly not automatable and have no suitable candidates.

Example output:
[
  {{
    "node_id": "n1",
    "is_automatic": true,
    "confidence": 0.85,
    "selected_activity_id": "send_email_gmail",
    "reasoning": "Strong match with Gmail send activity. Task involves sending email which is fully automatable.",
  }},
  {{
    "node_id": "n2",
    "is_automatic": false,
    "confidence": 0.2,
    "selected_activity_id": null,
    "reasoning": "Task requires human judgment and approval. No suitable automation candidates.",
  }}
]

Now evaluate the nodes and return the JSON array."""
    
    try:
        json_prompt = prompt
        
        response = model.invoke(json_prompt)
        
        response_text = response.content if hasattr(response, 'content') else str(response)
        
        if hasattr(response, 'response_metadata'):
            print("Token used in automation feasibility evaluation:", response.response_metadata)
        
        # Clean response text
        response_text = response_text.strip()
        if response_text.startswith("```json"):
            response_text = response_text[7:]
        if response_text.startswith("```"):
            response_text = response_text[3:]
        if response_text.endswith("```"):
            response_text = response_text[:-3]
        response_text = response_text.strip()
        
        # Parse JSON
        evaluations = json.loads(response_text)
        
        # Ensure it's a list
        if isinstance(evaluations, dict):
            evaluations = [evaluations]
        elif not isinstance(evaluations, list):
            evaluations = []
        
        return evaluations
        
    except json.JSONDecodeError as e:
        print(f"⚠️ LLM returned invalid JSON for automation evaluation: {response_text}")
        print(f"JSON decode error: {e}")
        return []
    except Exception as e:
        print(f"⚠️ Error in automation feasibility evaluation: {e}")
        return []