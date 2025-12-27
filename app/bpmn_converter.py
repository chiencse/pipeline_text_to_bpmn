"""
bpmn_converter.py

Python port of json-to-bpmn-xml.util.ts
Converts structured JSON with BPMN nodes and flows into valid BPMN 2.0 XML
and generates activities list compatible with the RPA system.

Usage:
    from app.bpmn_converter import convert_json_to_process
    result = convert_json_to_process({"bpmn": {...}, "mapping": [...]})
    # result = {"success": True, "xml": "...", "activities": [...], "variables": [...]}
"""

from typing import Dict, Any, List, Optional, Tuple, Set
from dataclasses import dataclass, field
from collections import defaultdict
import html
import time

# =============================================================================
# TYPE DEFINITIONS
# =============================================================================

@dataclass
class BpmnNodeMapping:
    is_automatic: bool = False
    bot_id: Optional[str] = None
    manual_review: bool = False


@dataclass
class BpmnNodeJson:
    id: str
    type: str
    name: str
    mapping: Optional[BpmnNodeMapping] = None
    lane: Optional[str] = None
    pool: Optional[str] = None


@dataclass
class BpmnFlowJson:
    source: str
    target: str
    type: str = "SequenceFlow"
    condition: Optional[str] = None


@dataclass
class ActivityMappingCandidate:
    activity_id: str
    score: float


@dataclass
class ActivityMapping:
    node_id: str
    activity_id: Optional[str] = None
    confidence: Optional[float] = None
    manual_review: bool = False
    type: str = ""
    candidates: List[Dict[str, Any]] = field(default_factory=list)
    input_bindings: Dict[str, Any] = field(default_factory=dict)
    outputs: List[Any] = field(default_factory=list)


@dataclass
class Activity:
    activityID: str
    activityName: str
    activityType: str
    keyword: str = ""
    properties: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Variable:
    name: str
    type: str
    value: Any = None
    description: str = ""


@dataclass
class NodePosition:
    x: float
    y: float


@dataclass
class LayoutResult:
    positions: Dict[str, NodePosition]
    waypoints: Dict[str, List[Dict[str, float]]]


@dataclass
class LayoutOptions:
    horizontal_spacing: int = 180
    vertical_spacing: int = 120
    start_x: int = 200
    start_y: int = 200
    branch_spacing: int = 140


@dataclass
class ProcessConversionResult:
    success: bool
    xml: Optional[str] = None
    activities: Optional[List[Dict[str, Any]]] = None
    variables: Optional[List[Dict[str, Any]]] = None
    errors: Optional[List[str]] = None


# Shape dimensions for different BPMN element types
SHAPE_DIMENSIONS: Dict[str, Dict[str, int]] = {
    "StartEvent": {"width": 36, "height": 36},
    "EndEvent": {"width": 36, "height": 36},
    "Task": {"width": 100, "height": 80},
    "UserTask": {"width": 100, "height": 80},
    "ServiceTask": {"width": 100, "height": 80},
    "ManualTask": {"width": 100, "height": 80},
    "SendTask": {"width": 100, "height": 80},
    "ReceiveTask": {"width": 100, "height": 80},
    "ScriptTask": {"width": 100, "height": 80},
    "BusinessRuleTask": {"width": 100, "height": 80},
    "ExclusiveGateway": {"width": 50, "height": 50},
    "ParallelGateway": {"width": 50, "height": 50},
    "InclusiveGateway": {"width": 50, "height": 50},
    "Gateway": {"width": 50, "height": 50},
    "SubProcess": {"width": 350, "height": 200},
    "DataObject": {"width": 36, "height": 50},
}

# BPMN element type mapping
BPMN_ELEMENT_MAP: Dict[str, str] = {
    "StartEvent": "bpmn:startEvent",
    "EndEvent": "bpmn:endEvent",
    "Task": "bpmn:task",
    "UserTask": "bpmn:userTask",
    "ServiceTask": "bpmn:serviceTask",
    "ManualTask": "bpmn:manualTask",
    "SendTask": "bpmn:sendTask",
    "ReceiveTask": "bpmn:receiveTask",
    "ScriptTask": "bpmn:scriptTask",
    "BusinessRuleTask": "bpmn:businessRuleTask",
    "ExclusiveGateway": "bpmn:exclusiveGateway",
    "ParallelGateway": "bpmn:parallelGateway",
    "InclusiveGateway": "bpmn:inclusiveGateway",
    "Gateway": "bpmn:exclusiveGateway",
    "SubProcess": "bpmn:subProcess",
    "DataObject": "bpmn:dataObjectReference",
}


# =============================================================================
# IMPROVED LAYOUT ALGORITHM (Sugiyama-style)
# =============================================================================

def calculate_layout(
    nodes: List[Dict[str, Any]],
    flows: List[Dict[str, Any]],
    options: Optional[LayoutOptions] = None
) -> LayoutResult:
    """
    Improved auto-layout algorithm with better branch handling.
    Uses Sugiyama-style layer assignment and handles gateway branching with proper Y-offset.
    """
    opts = options or LayoutOptions()
    positions: Dict[str, NodePosition] = {}
    waypoints: Dict[str, List[Dict[str, float]]] = {}

    if not nodes:
        return LayoutResult(positions={}, waypoints={})

    # Build adjacency lists
    outgoing: Dict[str, List[str]] = defaultdict(list)
    incoming: Dict[str, List[str]] = defaultdict(list)
    in_degree: Dict[str, int] = defaultdict(int)
    out_degree: Dict[str, int] = defaultdict(int)

    for node in nodes:
        node_id = node.get("id", "")
        outgoing[node_id] = []
        incoming[node_id] = []
        in_degree[node_id] = 0
        out_degree[node_id] = 0

    for flow in flows:
        source = flow.get("source", "")
        target = flow.get("target", "")
        if source and target:
            outgoing[source].append(target)
            incoming[target].append(source)
            in_degree[target] += 1
            out_degree[source] += 1

    node_by_id = {n.get("id", ""): n for n in nodes}

    # Identify gateways (split/join points)
    split_gateways: Set[str] = set()
    join_gateways: Set[str] = set()

    for node in nodes:
        node_id = node.get("id", "")
        node_type = node.get("type", "")
        if "Gateway" in node_type:
            if out_degree[node_id] > 1:
                split_gateways.add(node_id)
            if in_degree[node_id] > 1:
                join_gateways.add(node_id)

    # Layer assignment using BFS from start nodes
    layers: List[List[str]] = []
    node_layer: Dict[str, int] = {}
    visited: Set[str] = set()

    # Find start nodes (no incoming edges)
    current_layer = [n.get("id", "") for n in nodes if in_degree.get(n.get("id", ""), 0) == 0]

    layer_index = 0
    while current_layer:
        layers.append(list(current_layer))
        for node_id in current_layer:
            visited.add(node_id)
            node_layer[node_id] = layer_index

        next_layer: List[str] = []
        for node_id in current_layer:
            for target_id in outgoing.get(node_id, []):
                if target_id not in visited and target_id not in next_layer:
                    is_join = target_id in join_gateways
                    if is_join:
                        all_predecessors_visited = all(
                            pred in visited for pred in incoming.get(target_id, [])
                        )
                        if all_predecessors_visited:
                            next_layer.append(target_id)
                    else:
                        next_layer.append(target_id)

        current_layer = next_layer
        layer_index += 1

        # Safety check to prevent infinite loops
        if layer_index > len(nodes) * 2:
            break

    # Handle remaining nodes (cycles or disconnected)
    for node in nodes:
        node_id = node.get("id", "")
        if node_id not in visited:
            preds = incoming.get(node_id, [])
            max_pred_layer = -1
            for pred in preds:
                pred_layer = node_layer.get(pred)
                if pred_layer is not None and pred_layer > max_pred_layer:
                    max_pred_layer = pred_layer

            target_layer = max_pred_layer + 1
            while len(layers) <= target_layer:
                layers.append([])
            layers[target_layer].append(node_id)
            node_layer[node_id] = target_layer
            visited.add(node_id)

    # Track branch paths for Y-positioning
    node_branch_index: Dict[str, int] = {}

    # Trace branches from split gateways
    for gateway_id in split_gateways:
        targets = outgoing.get(gateway_id, [])
        for branch_idx, target_id in enumerate(targets):
            queue = [target_id]
            branch_visited: Set[str] = set()

            while queue:
                current_node = queue.pop(0)
                if current_node in branch_visited or current_node in join_gateways:
                    continue
                branch_visited.add(current_node)

                if current_node not in node_branch_index:
                    node_branch_index[current_node] = branch_idx

                for next_node in outgoing.get(current_node, []):
                    if next_node not in branch_visited:
                        queue.append(next_node)

    # Calculate positions with improved Y-spacing
    for layer_idx, layer in enumerate(layers):
        # Sort nodes in layer by branch index
        layer.sort(key=lambda nid: node_branch_index.get(nid, 0))

        layer_height = len(layer) * opts.branch_spacing
        start_y = opts.start_y - layer_height / 2 + opts.branch_spacing / 2

        for node_idx, node_id in enumerate(layer):
            node = node_by_id.get(node_id)
            if not node:
                continue

            x = opts.start_x + layer_idx * opts.horizontal_spacing

            branch_idx = node_branch_index.get(node_id)
            if branch_idx is not None and len(layer) > 1:
                y = opts.start_y + branch_idx * opts.branch_spacing
            elif len(layer) == 1:
                y = opts.start_y
            else:
                y = start_y + node_idx * opts.branch_spacing

            positions[node_id] = NodePosition(x=x, y=y)

    # Ensure join gateways are centered between their incoming branches
    for gateway_id in join_gateways:
        preds = incoming.get(gateway_id, [])
        if len(preds) > 1:
            min_y = float('inf')
            max_y = float('-inf')

            for pred_id in preds:
                pred_pos = positions.get(pred_id)
                if pred_pos:
                    min_y = min(min_y, pred_pos.y)
                    max_y = max(max_y, pred_pos.y)

            gateway_pos = positions.get(gateway_id)
            if gateway_pos and min_y != float('inf'):
                gateway_pos.y = (min_y + max_y) / 2

    # Calculate waypoints for flows
    for flow in flows:
        source_id = flow.get("source", "")
        target_id = flow.get("target", "")
        source_node = node_by_id.get(source_id)
        target_node = node_by_id.get(target_id)
        source_pos = positions.get(source_id)
        target_pos = positions.get(target_id)

        if not all([source_node, target_node, source_pos, target_pos]):
            continue

        source_dims = SHAPE_DIMENSIONS.get(source_node.get("type", "Task"), {"width": 100, "height": 80})
        target_dims = SHAPE_DIMENSIONS.get(target_node.get("type", "Task"), {"width": 100, "height": 80})

        start_x = source_pos.x + source_dims["width"]
        start_y = source_pos.y + source_dims["height"] / 2
        end_x = target_pos.x
        end_y = target_pos.y + target_dims["height"] / 2

        flow_waypoints: List[Dict[str, float]] = []

        is_from_split = source_id in split_gateways
        is_to_join = target_id in join_gateways
        needs_routing = abs(end_y - start_y) > 30 or end_x <= start_x

        if is_from_split and needs_routing:
            mid_x = start_x + 40
            flow_waypoints = [
                {"x": start_x, "y": start_y},
                {"x": mid_x, "y": start_y},
                {"x": mid_x, "y": end_y},
                {"x": end_x, "y": end_y},
            ]
        elif is_to_join and needs_routing:
            mid_x = end_x - 40
            flow_waypoints = [
                {"x": start_x, "y": start_y},
                {"x": mid_x, "y": start_y},
                {"x": mid_x, "y": end_y},
                {"x": end_x, "y": end_y},
            ]
        elif needs_routing:
            mid_x = (start_x + end_x) / 2
            flow_waypoints = [
                {"x": start_x, "y": start_y},
                {"x": mid_x, "y": start_y},
                {"x": mid_x, "y": end_y},
                {"x": end_x, "y": end_y},
            ]
        else:
            flow_waypoints = [
                {"x": start_x, "y": start_y},
                {"x": end_x, "y": end_y},
            ]

        flow_id = f"Flow_{source_id}_{target_id}"
        waypoints[flow_id] = flow_waypoints

    return LayoutResult(positions=positions, waypoints=waypoints)


# =============================================================================
# XML GENERATION
# =============================================================================

def escape_xml(s: str) -> str:
    """Escape XML special characters."""
    return html.escape(s, quote=True)


def generate_process_id() -> str:
    """Generate a unique process ID."""
    return f"Process_{int(time.time() * 1000):x}"


def get_bpmn_element_name(node_type: str) -> str:
    """Get the BPMN element name for a node type."""
    return BPMN_ELEMENT_MAP.get(node_type, "bpmn:task")


def generate_flow_id(source: str, target: str) -> str:
    """Generate a flow ID from source and target."""
    return f"Flow_{source}_{target}"


def build_flow_references(
    node_id: str,
    flows: List[Dict[str, Any]]
) -> Tuple[List[str], List[str]]:
    """Build incoming and outgoing flow references for a node."""
    incoming: List[str] = []
    outgoing: List[str] = []

    for flow in flows:
        flow_id = generate_flow_id(flow.get("source", ""), flow.get("target", ""))
        if flow.get("source") == node_id:
            outgoing.append(flow_id)
        if flow.get("target") == node_id:
            incoming.append(flow_id)

    return incoming, outgoing


def json_to_bpmn_xml(
    data: Dict[str, Any],
    layout_options: Optional[LayoutOptions] = None
) -> str:
    """
    Main function to convert JSON to BPMN XML.
    
    Args:
        data: Dict with structure {"bpmn": {"nodes": [...], "flows": [...]}, "mapping": [...]}
        layout_options: Optional layout configuration
        
    Returns:
        BPMN 2.0 XML string
    """
    bpmn = data.get("bpmn", {})
    nodes = bpmn.get("nodes", [])
    flows = bpmn.get("flows", [])
    mapping = data.get("mapping", [])
    process_id = generate_process_id()
    definitions_id = f"Definitions_{int(time.time() * 1000):x}"
    diagram_id = f"BPMNDiagram_{int(time.time() * 1000):x}"
    plane_id = f"BPMNPlane_{int(time.time() * 1000):x}"

    # Calculate layout
    layout = calculate_layout(nodes, flows, layout_options)

    # Build the BPMN XML
    xml_parts = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        '<bpmn:definitions',
        '  xmlns:bpmn="http://www.omg.org/spec/BPMN/20100524/MODEL"',
        '  xmlns:bpmndi="http://www.omg.org/spec/BPMN/20100524/DI"',
        '  xmlns:dc="http://www.omg.org/spec/DD/20100524/DC"',
        '  xmlns:di="http://www.omg.org/spec/DD/20100524/DI"',
        '  xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"',
        '  xmlns:bioc="http://bpmn.io/schema/bpmn/biocolor/1.0"',
        f'  id="{definitions_id}"',
        '  targetNamespace="http://bpmn.io/schema/bpmn"',
        '  exporter="edu-rpa"',
        '  exporterVersion="1.0">',
        '',
        f'  <bpmn:process id="{process_id}" isExecutable="true">',
    ]

    # Generate nodes
    for node in nodes:
        node_id = node.get("id", "")
        node_type = node.get("type", "Task")
        node_name = node.get("name", "")
        
        element_name = get_bpmn_element_name(node_type)
        incoming, outgoing = build_flow_references(node_id, flows)
        name_attr = f' name="{escape_xml(node_name)}"' if node_name else ""

        xml_parts.append(f'    <{element_name} id="{node_id}"{name_attr}>')

        for flow_id in incoming:
            xml_parts.append(f'      <bpmn:incoming>{flow_id}</bpmn:incoming>')

        for flow_id in outgoing:
            xml_parts.append(f'      <bpmn:outgoing>{flow_id}</bpmn:outgoing>')

        xml_parts.append(f'    </{element_name}>')

    # Generate sequence flows
    for flow in flows:
        source = flow.get("source", "")
        target = flow.get("target", "")
        condition = flow.get("condition")
        flow_id = generate_flow_id(source, target)
        name_attr = f' name="{escape_xml(condition)}"' if condition else ""

        if condition:
            xml_parts.append(
                f'    <bpmn:sequenceFlow id="{flow_id}" sourceRef="{source}" targetRef="{target}"{name_attr}>'
            )
            xml_parts.append(
                f'      <bpmn:conditionExpression xsi:type="bpmn:tFormalExpression">{escape_xml(condition)}</bpmn:conditionExpression>'
            )
            xml_parts.append('    </bpmn:sequenceFlow>')
        else:
            xml_parts.append(
                f'    <bpmn:sequenceFlow id="{flow_id}" sourceRef="{source}" targetRef="{target}"{name_attr} />'
            )

    xml_parts.append('  </bpmn:process>')
    xml_parts.append('')

    # Generate BPMN Diagram
    xml_parts.append(f'  <bpmndi:BPMNDiagram id="{diagram_id}">')
    xml_parts.append(f'    <bpmndi:BPMNPlane id="{plane_id}" bpmnElement="{process_id}">')

    # Build mapping dict for quick lookup by node_id
    # Mapping can be: list of dicts with "node_id" key, or list of dicts like {node_id: {...}}
    mapping_dict = {}
    if isinstance(mapping, list):
        for m in mapping:
            if isinstance(m, dict):
                # Check if it's {node_id: mapping_entry} format
                if len(m) == 1:
                    node_id = list(m.keys())[0]
                    mapping_dict[node_id] = m[node_id]
                # Or direct format with "node_id" key
                elif "node_id" in m:
                    mapping_dict[m["node_id"]] = m
    elif isinstance(mapping, dict):
        mapping_dict = mapping

    # Generate shapes
    for node in nodes:
        node_id = node.get("id", "")
        node_type = node.get("type", "Task")
        node_name = node.get("name", "")
        # Get is_automatic from mapping dict
        node_mapping = mapping_dict.get(node_id, {})
        is_automatic = node_mapping.get("is_automatic", False)
        pos = layout.positions.get(node_id)
        if not pos:
            continue

        dims = SHAPE_DIMENSIONS.get(node_type, {"width": 100, "height": 80})
        shape_id = f"{node_id}_di"

        xml_parts.append(f'      <bpmndi:BPMNShape id="{shape_id}" bpmnElement="{node_id}">')
        xml_parts.append(
            f'        <dc:Bounds x="{int(pos.x)}" y="{int(pos.y)}" width="{dims["width"]}" height="{dims["height"]}" />'
        )
        if is_automatic:
            # Màu xanh dương cho các node tự động
            xml_parts.append(
                '        <bioc:ColoredShape fill="#e3f2fd" stroke="#2196f3" />'
            )
        # Add label for gateways and events
        if node_name and node_type in ["ExclusiveGateway", "ParallelGateway", "InclusiveGateway", "Gateway", "StartEvent", "EndEvent"]:
            xml_parts.append('        <bpmndi:BPMNLabel>')
            xml_parts.append(
                f'          <dc:Bounds x="{int(pos.x - 10)}" y="{int(pos.y + dims["height"] + 5)}" width="{dims["width"] + 20}" height="14" />'
            )
            xml_parts.append('        </bpmndi:BPMNLabel>')

        xml_parts.append('      </bpmndi:BPMNShape>')

    # Generate edges
    for flow in flows:
        source = flow.get("source", "")
        target = flow.get("target", "")
        condition = flow.get("condition")
        flow_id = generate_flow_id(source, target)
        points = layout.waypoints.get(flow_id, [])
        edge_id = f"{flow_id}_di"

        xml_parts.append(f'      <bpmndi:BPMNEdge id="{edge_id}" bpmnElement="{flow_id}">')

        for point in points:
            xml_parts.append(
                f'        <di:waypoint x="{round(point["x"])}" y="{round(point["y"])}" />'
            )

        if condition and points:
            mid_x = (points[0]["x"] + points[-1]["x"]) / 2
            mid_y = (points[0]["y"] + points[-1]["y"]) / 2 - 15
            xml_parts.append('        <bpmndi:BPMNLabel>')
            xml_parts.append(
                f'          <dc:Bounds x="{round(mid_x)}" y="{round(mid_y)}" width="40" height="14" />'
            )
            xml_parts.append('        </bpmndi:BPMNLabel>')

        xml_parts.append('      </bpmndi:BPMNEdge>')

    xml_parts.append('    </bpmndi:BPMNPlane>')
    xml_parts.append('  </bpmndi:BPMNDiagram>')
    xml_parts.append('</bpmn:definitions>')

    return '\n'.join(xml_parts)


# =============================================================================
# ACTIVITY GENERATION FROM MAPPING
# =============================================================================

def build_activity_properties(
    node: Dict[str, Any],
    mapping: Dict[str, Any]
) -> Dict[str, Any]:
    """Build activity properties from mapping data."""
    activity_id = mapping.get("activity_id", "") or ""
    
    # Parse activity_id to extract package and activity name
    activity_parts = activity_id.split(".")
    has_package = len(activity_parts) > 1
    activity_package = activity_parts[0] if has_package else ""
    activity_name = ".".join(activity_parts[1:]) if has_package else activity_id

    properties: Dict[str, Any] = {
        "activityPackage": activity_package,
        "activityName": activity_name,
        "serviceName": activity_package,
        "arguments": {},
        "assigns": [],
        "_mapping": {
            "confidence": mapping.get("confidence"),
            "manual_review": mapping.get("manual_review", False),
            "candidates": mapping.get("candidates", []),
            "input_bindings": mapping.get("input_bindings", {}),
            "outputs": mapping.get("outputs", []),
        },
    }

    # Convert input_bindings to arguments
    input_bindings = mapping.get("input_bindings", {})
    if isinstance(input_bindings, dict):
        args: Dict[str, Dict[str, Any]] = {}
        for key, value in input_bindings.items():
            args[key] = {
                "type": "string",
                "description": "",
                "keywordArg": key,
                "value": str(value) if value is not None else "",
                "overrideType": None,
            }
        properties["arguments"] = args

    return properties


def generate_activities(
    nodes: List[Dict[str, Any]],
    flows: List[Dict[str, Any]],
    mappings: Optional[List[Dict[str, Any]]] = None
) -> List[Dict[str, Any]]:
    """
    Generate activities list from BPMN nodes and activity mappings.
    Creates the activities array compatible with the RPA system.
    """
    mapping_by_node_id: Dict[str, Dict[str, Any]] = {}
    if mappings:
        for m in mappings:
            node_id = m.get("node_id")
            if node_id:
                mapping_by_node_id[node_id] = m

    activities: List[Dict[str, Any]] = []

    # Generate activities for all nodes
    for node in nodes:
        node_id = node.get("id", "")
        node_type = node.get("type", "Task")
        node_name = node.get("name", "")
        
        bpmn_type = f"bpmn:{node_type[0].lower()}{node_type[1:]}"
        mapping = mapping_by_node_id.get(node_id)

        activity: Dict[str, Any] = {
            "activityID": node_id,
            "activityName": node_name,
            "activityType": bpmn_type,
            "keyword": mapping.get("activity_id", "") if mapping else "",
            "properties": {},
        }

        # If mapping exists, build properties from it
        if mapping:
            activity["properties"] = build_activity_properties(node, mapping)

        activities.append(activity)

    # Generate activities for flows (for condition flows)
    for flow in flows:
        condition = flow.get("condition")
        if condition:
            source = flow.get("source", "")
            target = flow.get("target", "")
            flow_id = generate_flow_id(source, target)
            activities.append({
                "activityID": flow_id,
                "activityName": condition,
                "activityType": "bpmn:sequenceFlow",
                "properties": {
                    "arguments": {},
                },
            })

    return activities


# =============================================================================
# VALIDATION
# =============================================================================

def validate_bpmn_json(data: Any) -> Tuple[bool, List[str]]:
    """
    Validate BPMN JSON structure.
    
    Returns:
        Tuple of (is_valid, error_messages)
    """
    errors: List[str] = []

    if not data or not isinstance(data, dict):
        errors.append("Input must be an object")
        return False, errors

    bpmn = data.get("bpmn")
    if not bpmn or not isinstance(bpmn, dict):
        errors.append('Missing or invalid "bpmn" property')
        return False, errors

    nodes = bpmn.get("nodes")
    flows = bpmn.get("flows")

    if not isinstance(nodes, list):
        errors.append('Missing or invalid "bpmn.nodes" array')

    if not isinstance(flows, list):
        errors.append('Missing or invalid "bpmn.flows" array')

    if errors:
        return False, errors

    # Validate nodes
    node_ids: Set[str] = set()
    for index, node in enumerate(nodes):
        if not isinstance(node, dict):
            errors.append(f"Node at index {index} is not an object")
            continue
            
        node_id = node.get("id")
        if not node_id:
            errors.append(f'Node at index {index} is missing "id"')
        elif node_id in node_ids:
            errors.append(f"Duplicate node id: {node_id}")
        else:
            node_ids.add(node_id)

        if not node.get("type"):
            errors.append(f'Node at index {index} is missing "type"')

    # Validate flows
    for index, flow in enumerate(flows):
        if not isinstance(flow, dict):
            errors.append(f"Flow at index {index} is not an object")
            continue
            
        source = flow.get("source")
        target = flow.get("target")

        if not source:
            errors.append(f'Flow at index {index} is missing "source"')
        elif source not in node_ids:
            errors.append(f'Flow at index {index} references non-existent source node: {source}')

        if not target:
            errors.append(f'Flow at index {index} is missing "target"')
        elif target not in node_ids:
            errors.append(f'Flow at index {index} references non-existent target node: {target}')

    # Check for start and end events
    has_start_event = any(n.get("type") == "StartEvent" for n in nodes if isinstance(n, dict))
    has_end_event = any(n.get("type") == "EndEvent" for n in nodes if isinstance(n, dict))

    if not has_start_event:
        errors.append("Process must have at least one StartEvent")

    if not has_end_event:
        errors.append("Process must have at least one EndEvent")

    return len(errors) == 0, errors


# =============================================================================
# COMPLETE CONVERTER
# =============================================================================

def convert_json_to_process(
    data: Any,
    layout_options: Optional[LayoutOptions] = None
) -> ProcessConversionResult:
    """
    Complete converter that takes JSON and returns all necessary data for RPA system.
    
    Args:
        data: Dict with structure {"bpmn": {"nodes": [...], "flows": [...]}, "mapping": [...]}
        layout_options: Optional layout configuration
        
    Returns:
        ProcessConversionResult with:
        - success: bool
        - xml: BPMN diagram XML for visualization
        - activities: List of activities with properties for each node
        - variables: Placeholder for variables (to be populated later)
        - errors: List of error messages if failed
    """
    # Validate input
    is_valid, errors = validate_bpmn_json(data)
    if not is_valid:
        return ProcessConversionResult(success=False, errors=errors)

    try:
        # Generate XML
        xml = json_to_bpmn_xml(data, layout_options)

        # Generate activities from nodes and mappings
        bpmn = data.get("bpmn", {})
        nodes = bpmn.get("nodes", [])
        flows = bpmn.get("flows", [])
        mappings = data.get("mapping", [])

        activities = generate_activities(nodes, flows, mappings)

        # Initialize empty variables (to be populated by user later)
        variables: List[Dict[str, Any]] = []

        return ProcessConversionResult(
            success=True,
            xml=xml,
            activities=activities,
            variables=variables,
        )
    except Exception as e:
        return ProcessConversionResult(
            success=False,
            errors=[f"Conversion failed: {str(e)}"],
        )


def convert_json_to_bpmn(data: Any) -> Dict[str, Any]:
    """
    Convert JSON to BPMN XML with validation (legacy function compatibility).
    
    Returns:
        Dict with {"success": bool, "xml"?: str, "errors"?: List[str]}
    """
    is_valid, errors = validate_bpmn_json(data)

    if not is_valid:
        return {"success": False, "errors": errors}

    try:
        xml = json_to_bpmn_xml(data)
        return {"success": True, "xml": xml}
    except Exception as e:
        return {"success": False, "errors": [f"XML generation failed: {str(e)}"]}


# =============================================================================
# EXPORTS FOR NODE_RENDER
# =============================================================================

def render_bpmn_output(
    bpmn: Dict[str, Any],
    mapping: Optional[List[Dict[str, Any]]] = None,
    layout_options: Optional[LayoutOptions] = None
) -> Dict[str, Any]:
    """
    Helper function for node_render to generate all output data.
    
    Args:
        bpmn: Dict with {"nodes": [...], "flows": [...]}
        mapping: Optional list of activity mappings
        layout_options: Optional layout configuration
        
    Returns:
        Dict with:
        - success: bool
        - xml: BPMN XML string
        - activities: List of activity dicts
        - variables: List of variable dicts (empty for now)
        - errors: List of error messages if failed
    """
    data = {
        "bpmn": bpmn,
        "mapping": mapping or [],
    }
    
    result = convert_json_to_process(data, layout_options)
    
    return {
        "success": result.success,
        "xml": result.xml,
        "activities": result.activities,
        "variables": result.variables,
        "errors": result.errors,
    }


# Test function
if __name__ == "__main__":
    sample_data = {
        "bpmn": {
            "nodes": [
                {"id": "start_1", "type": "StartEvent", "name": "Start"},
                {"id": "task_1", "type": "UserTask", "name": "Send Email"},
                {"id": "gateway_1", "type": "ExclusiveGateway", "name": "Approved?"},
                {"id": "task_2", "type": "ServiceTask", "name": "Update SAP"},
                {"id": "task_3", "type": "ManualTask", "name": "Manual Review"},
                {"id": "end_1", "type": "EndEvent", "name": "End"},
            ],
            "flows": [
                {"source": "start_1", "target": "task_1", "type": "SequenceFlow"},
                {"source": "task_1", "target": "gateway_1", "type": "SequenceFlow"},
                {"source": "gateway_1", "target": "task_2", "type": "SequenceFlow", "condition": "approved == true"},
                {"source": "gateway_1", "target": "task_3", "type": "SequenceFlow", "condition": "approved == false"},
                {"source": "task_2", "target": "end_1", "type": "SequenceFlow"},
                {"source": "task_3", "target": "end_1", "type": "SequenceFlow"},
            ],
        },
        "mapping": [
            {
                "node_id": "task_1",
                "activity_id": "gmail.send_email",
                "confidence": 0.85,
                "manual_review": False,
                "type": "ServiceTask",
                "candidates": [{"activity_id": "gmail.send_email", "score": 0.85}],
                "input_bindings": {"to": "finance@company.com", "subject": "Quotation"},
                "outputs": [],
            },
            {
                "node_id": "task_2",
                "activity_id": "sap.update_invoice",
                "confidence": 0.78,
                "manual_review": True,
                "type": "ServiceTask",
                "candidates": [{"activity_id": "sap.update_invoice", "score": 0.78}],
                "input_bindings": {},
                "outputs": [],
            },
        ],
    }

    result = convert_json_to_process(sample_data)
    
    if result.success:
        print("=== XML ===")
        print(result.xml[:500] + "..." if result.xml and len(result.xml) > 500 else result.xml)
        print("\n=== ACTIVITIES ===")
        for act in (result.activities or []):
            print(f"  - {act['activityID']}: {act['activityName']} ({act['activityType']})")
        print("\n=== VARIABLES ===")
        print(f"  {result.variables}")
    else:
        print("ERRORS:", result.errors)

