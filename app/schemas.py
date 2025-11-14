# app/schemas.py
from pydantic import BaseModel, Field
from typing import List, Optional, Literal, Dict, Any

# Updated entity labels for BPMN/ERP domain
EntityLabel = Literal[
    "Action",
    "Object",
    "Role",
    "System",
    "Event",
    "DataField",
    "Condition",
    "Resource",
    "Process",
    "Output"
]

# Relation types kept (adjust if you need more)
RelType = Literal[
    "agent_of",       # Subject performs an action
    "object_of",      # Action targets an object
    "receiver_of",    # Action directed to someone
    "exec_in",        # Location of action
    "exec_at",        # Time of action
    "sequence_next",  # Sequential relation between actions
    "condition_of",   # Conditional relation (if ... then)
    "use_of",         # Instrument/tool used
    "purpose_of",     # Purpose or reason
    "result_of"       # Output or effect of an action
]

# Flow and Node types kept
FlowType = Literal["SequenceFlow", "MessageFlow", "DataAssociation"]
NodeType = Literal["StartEvent", "EndEvent", "Task", "ManualTask", "UserTask", "Gateway", "DataObject"]

class Entity(BaseModel):
    # allow confidence and optional offsets
    label: EntityLabel
    text: str
    start: Optional[int] = None
    end: Optional[int] = None
    confidence: Optional[float] = None

class Relation(BaseModel):
    head: str
    tail: str
    type: RelType

class BPMNNode(BaseModel):
    id: str
    type: NodeType
    name: str
    lane: Optional[str] = None
    pool: Optional[str] = None
    subprocess: Optional[str] = None  # e.g., "bot1", "human_subprocess"
    attributes: Dict[str, Any] = Field(default_factory=dict)

class BPMNFlow(BaseModel):
    source: str
    target: str
    type: FlowType
    condition: Optional[str] = None

class MappingCandidate(BaseModel):
    activity_id: str
    score: float
    confidence: Optional[float] = None
    manual_review: bool = False
    pkg: Optional[str] = None
    keyword: Optional[str] = None

class Mapping(BaseModel):
    node_id: str
    activity_id: Optional[str] = None
    confidence: Optional[float] = None
    manual_review: bool = False
    bot_id: Optional[str] = None  
    candidates: List[MappingCandidate] = Field(default_factory=list)
    input_bindings: Dict[str, Any] = Field(default_factory=dict)
    outputs: List[str] = Field(default_factory=list)

class GraphOutput(BaseModel):
    intent: Optional[Dict[str, Any]] = None
    entities: List[Entity] = Field(default_factory=list)
    relations: List[Relation] = Field(default_factory=list)
    bpmn: Dict[str, List[Any]]  # {"nodes":[...], "flows":[...]}
    mapping: Optional[List[Mapping]] = None
