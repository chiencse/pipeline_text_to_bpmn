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
def call_llm_ner(chunks: List[str]) -> List[Dict]:
    """
    Gọi Gemini để thực hiện NER (Named Entity Recognition)
    """
    prompt = f"""
    You are an information extraction model.
    Extract entities (actions, objects, roles, systems) from the following text chunks.
    Return JSON list only, each entity as:
      {{"text": "...", "label": "...", "confidence": 0.0}}
    Text chunks: {chunks}
    """

    model = genai.GenerativeModel(MODEL_NER)
    response = model.generate_content(prompt, generation_config={"response_mime_type": "application/json"})

    try:
        entities = response.text
        import json
        return json.loads(entities)
    except Exception:
        print("⚠️ Gemini returned non-JSON, raw output:", response.text)
        return []

# ========== CALL BPMN WITH MAPPING ==========
def call_llm_bpmn_with_mapping(text: str, entities: List[Dict], candidates: List[Dict]) -> GraphOutput:
    """
    Gọi Gemini để sinh BPMN flow + mapping.
    """
    model = genai.GenerativeModel(MODEL_BPMN)

    prompt = f"""
    You are a process modeling assistant to mapping candidate activities to BPMN Process. Your task is to generate a BPMN process model and map candidate activities to the BPMN nodes.
    If candidate activities is not proper with context of the Input text, you can ignore them. Do not need use all candidate activities.
    Input text describes request of user about business process.
    Entities is a list of extracted entities from the text.
    Candidate Activities is a list of candidate activities retrieved from the activity library.
    Entities: {entities} 
    Candidate Activities: {candidates}
    Please output only JSON in the following structure:
    {{
      "intent": {{"code": "string", "confidence": 0.0}},
      "entities": [...], // class Entity(BaseModel):
            label: EntityLabel // Label Follow EntityLabel = Literal["ACT","DOC","SYS","ROLE","ID"]
            text: str
            start: int
            end: int
      "relations": [{{"head":"...","tail":"...","type":"..."}}], //RelType = Literal["agent_of","exec_in","object_of","receiver_of","sequence_next"]
      "bpmn": {{
        "nodes": [{{"id":"...","type":"Task|StartEvent|EndEvent","name":"..."}}],
        "flows": [{{"source":"...","target":"...","type":"SequenceFlow"}}]
      }}, // class BPMNNode(BaseModel):
            id: str
            type: NodeType  //NodeType = Literal["StartEvent","EndEvent","Task","ManualTask","UserTask","Gateway","DataObject"]
            name: str
            lane: Optional[str] = None
            pool: Optional[str] = None
            attributes: Dict[str, Any] = Field(default_factory=dict)

            class BPMNFlow(BaseModel):
                source: str
                target: str
                type: FlowType //FlowType = Literal["SequenceFlow","MessageFlow","DataAssociation"]
      "mapping": [{{"node_id":"...","activity_id":"...","confidence":0.0,"manual_review":false, "candidates":[]}}]
       // class Mapping(BaseModel):
            node_id: str
            activity_id: Optional[str] = None
            confidence: Optional[float] = None
            manual_review: bool = False
            candidates: List[MappingCandidate] = Field(default_factory=list)
            input_bindings: Dict[str, Any] = Field(default_factory=dict)
            outputs: List[str] = Field(default_factory=list)
          class MappingCandidate(BaseModel):
            activity_id: str
            score: float
            confidence: Optional[float] = None
            manual_review: bool = False
    }}
    Text input: {text}
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
    Given the text: {text}
    and entities: {entities}
    and relations: {relations}
    Generate BPMN process model as JSON structure:
    {{
      "intent": {{"code": "string", "confidence": 0.0}},
      "bpmn": {{
        "nodes": [{{"id":"...","type":"...","name":"..."}}],
        "flows": [{{"source":"...","target":"...","type":"..."}}]
      }}
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
