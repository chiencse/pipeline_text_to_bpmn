# app/schemas.py
from pydantic import BaseModel, Field
from typing import List, Optional, Literal, Dict, Any

EntityLabel = Literal["ACT","DOC","SYS","ROLE","ID"]
RelType = Literal["agent_of","exec_in","object_of","receiver_of","sequence_next"]
FlowType = Literal["SequenceFlow","MessageFlow","DataAssociation"]
NodeType = Literal["StartEvent","EndEvent","Task","ManualTask","UserTask","Gateway","DataObject"]

class Entity(BaseModel):
    label: EntityLabel
    text: str
    start: int
    end: int

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
    attributes: Dict[str, Any] = Field(default_factory=dict)

class BPMNFlow(BaseModel):
    source: str
    target: str
    type: FlowType

class MappingCandidate(BaseModel):
    activity_id: str
    score: float
    confidence: Optional[float] = None
    manual_review: bool = False
    
    

class Mapping(BaseModel):
    node_id: str
    activity_id: Optional[str] = None
    confidence: Optional[float] = None
    manual_review: bool = False
    candidates: List[MappingCandidate] = Field(default_factory=list)
    input_bindings: Dict[str, Any] = Field(default_factory=dict)
    outputs: List[str] = Field(default_factory=list)

class GraphOutput(BaseModel):
    intent: Optional[Dict[str, Any]] = None
    entities: List[Entity] = Field(default_factory=list)
    relations: List[Relation] = Field(default_factory=list)
    bpmn: Dict[str, List[Any]]  # {"nodes":[...], "flows":[...]}
    mapping: Optional[List[Mapping]] = None
