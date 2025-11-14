"""
generate_bpmn.py

Function to convert a GraphOutput-like dict (matching the provided pydantic schema) into
a BPMN 2.0 XML string.

Usage:
    xml = generate_bpmn_xml(graph)

graph should be a dict with keys:
  - entities: list[...] (not required for XML but kept)
  - relations: list[...] (not required)
  - bpmn: {"nodes": [...], "flows": [...]}

Node example:
    {"id":"task_1", "type":"Task", "name":"Do something", "lane":"Lane A"}
Flow example:
    {"source":"task_1","target":"task_2","type":"SequenceFlow","condition":null}

The produced XML will include basic BPMNDiagram/BPMNPlane shapes (minimal) so many editors
like Camunda will accept it. This is intentionally simple and focuses on correctness of
process elements and sequence/message/data flows.

"""
from typing import Dict, Any, List, Optional
import xml.etree.ElementTree as ET
from xml.dom import minidom
import os

# BPMN namespaces
BPMN_NS = "http://www.omg.org/spec/BPMN/20100524/MODEL"
BPMNDI_NS = "http://www.omg.org/spec/BPMN/20100524/DI"
OMGDC_NS = "http://www.omg.org/spec/DD/20100524/DC"
OMGDI_NS = "http://www.omg.org/spec/DD/20100524/DI"

NSMAP = {
    'bpmn': BPMN_NS,
    'bpmndi': BPMNDI_NS,
    'omgdc': OMGDC_NS,
    'omgdi': OMGDI_NS,
}

for prefix, uri in NSMAP.items():
    ET.register_namespace(prefix if prefix != 'bpmn' else '', uri)


def _tag(local: str) -> str:
    """Return qualified tag for BPMN namespace (default namespace)."""
    return f"{{{BPMN_NS}}}{local}"


def prettify_xml(elem: ET.Element) -> str:
    """Return a pretty-printed XML string for the Element."""
    rough = ET.tostring(elem, 'utf-8')
    reparsed = minidom.parseString(rough)
    return reparsed.toprettyxml(indent='  ')


# Mapping from NodeType to BPMN element name
NODE_TYPE_TO_TAG = {
    'StartEvent': 'startEvent',
    'EndEvent': 'endEvent',
    'Task': 'task',
    'UserTask': 'userTask',
    'ManualTask': 'manualTask',
    'ExclusiveGateway': 'exclusiveGateway',
    'ParallelGateway': 'parallelGateway',
    'ServiceTask': 'serviceTask',
    'DataObject': 'dataObjectReference',
}

# Mapping for FlowType to tag
FLOW_TYPE_TO_TAG = {
    'SequenceFlow': 'sequenceFlow',
    'MessageFlow': 'messageFlow',
    'DataAssociation': 'assignment',  # data associations are usually modeled as associations/artefacts
}


def _make_element(parent: ET.Element, tag: str, attrib: Dict[str, str] = None, text: Optional[str] = None) -> ET.Element:
    attrib = attrib or {}
    el = ET.SubElement(parent, _tag(tag), attrib)
    if text:
        el.set('name', text)
    return el


def generate_bpmn_xml(graph: Dict[str, Any], process_id: str = 'process_1', process_name: str = 'Generated Process') -> str:
    """
    Generate a BPMN 2.0 XML string from a GraphOutput-like dict.

    Args:
        graph: dict matching GraphOutput (must contain graph['bpmn']['nodes'] and graph['bpmn']['flows']).
        process_id: id attribute for the bpmn:process element
        process_name: name attribute for the bpmn:process

    Returns:
        BPMN 2.0 XML string
    """
    bpmn_data = graph.get('bpmn', {})
    nodes: List[Dict[str, Any]] = bpmn_data.get('nodes', [])
    flows: List[Dict[str, Any]] = bpmn_data.get('flows', [])

    # Build root definitions element with namespaces
    definitions_attrib = {
        'id': f'{process_id}_definitions',
        'targetNamespace': 'http://example.com/bpmn',
    }
    definitions = ET.Element(_tag('definitions'), definitions_attrib)

    # Create process
    process_attrib = {'id': process_id, 'name': process_name, 'isExecutable': 'false'}
    process_el = ET.SubElement(definitions, _tag('process'), process_attrib)

    # Collect lanes
    lanes_map: Dict[str, List[str]] = {}
    for n in nodes:
        lane = n.get('lane')
        if lane:
            lanes_map.setdefault(lane, []).append(n['id'])

    if lanes_map:
        lane_set = ET.SubElement(process_el, _tag('laneSet'), {'id': f'{process_id}_laneSet'})
        lane_idx = 1
        for lane_name, member_ids in lanes_map.items():
            lane_id = f'{process_id}_lane_{lane_idx}'
            lane_el = ET.SubElement(lane_set, _tag('lane'), {'id': lane_id, 'name': lane_name})
            # flowNodeRef elements
            for mid in member_ids:
                ET.SubElement(lane_el, _tag('flowNodeRef')).text = mid
            lane_idx += 1

    # Create elements for nodes
    # Keep track of IDs created to ensure references exist for flows
    created_ids = set()

    for node in nodes:
        nid = node.get('id')
        ntype = node.get('type')
        name = node.get('name') or nid
        attrib = {'id': nid, 'name': name}

        tag = NODE_TYPE_TO_TAG.get(ntype, None)
        if tag is None:
            # fallback: create a generic task
            tag = 'task'

        node_el = ET.SubElement(process_el, _tag(tag), attrib)

        # If DataObject create a dataObject element in definitions and a reference in process
        if tag == 'dataObjectReference':
            # Add item definition or dataObject under root definitions
            data_obj_id = f'{nid}_data'
            ET.SubElement(definitions, _tag('dataObject'), {'id': data_obj_id, 'name': name})

        created_ids.add(nid)

    # Create flows (sequenceFlow/messageFlow etc.)
    for idx, flow in enumerate(flows, start=1):
        source = flow.get('source')
        target = flow.get('target')
        ftype = flow.get('type', 'SequenceFlow')
        condition = flow.get('condition')

        # Skip flows referencing unknown nodes (defensive)
        if source not in created_ids or target not in created_ids:
            # Still create a sequenceFlow but keep it -- or skip. We'll skip but log as comment via attribute
            # add a note attribute
            attrib = {
                'id': f'{process_id}_flow_{idx}',
                'sourceRef': source or 'UNKNOWN',
                'targetRef': target or 'UNKNOWN',
            }
            attrib['bpmn:note'] = 'skipped_missing_target_or_source'
            ET.SubElement(process_el, _tag('sequenceFlow'), attrib)
            continue

        flow_tag = FLOW_TYPE_TO_TAG.get(ftype, 'sequenceFlow')
        flow_id = f'{process_id}_flow_{idx}'
        attrib = {'id': flow_id, 'sourceRef': source, 'targetRef': target}

        if flow_tag == 'sequenceFlow':
            f_el = ET.SubElement(process_el, _tag('sequenceFlow'), attrib)
            if condition:
                cond_el = ET.SubElement(f_el, _tag('conditionExpression'), {'xsi:type': 'tFormalExpression'})
                cond_el.text = condition
        elif flow_tag == 'messageFlow':
            # message flows are normally under definitions (between participants) but we'll put a minimal element
            ET.SubElement(definitions, _tag('messageFlow'), {'id': flow_id, 'sourceRef': source, 'targetRef': target})
        else:
            # fallback: sequenceFlow
            ET.SubElement(process_el, _tag('sequenceFlow'), attrib)

    # Optionally add a BPMNDiagram so editors render something. We will add a minimal diagram.
    bpmndi = ET.SubElement(definitions, f'{{{BPMNDI_NS}}}BPMNDiagram', {'id': f'{process_id}_BPMNDiagram'})
    plane = ET.SubElement(bpmndi, f'{{{BPMNDI_NS}}}BPMNPlane', {'id': f'{process_id}_BPMNPlane', 'bpmnElement': process_id})

    # For each node create a shape element (minimal bounds)
    x = 100
    y = 100
    dx = 200
    dy = 80
    for i, node in enumerate(nodes):
        nid = node['id']
        shape_attrib = {'id': f'{nid}_di', 'bpmnElement': nid}
        shape = ET.SubElement(plane, f'{{{BPMNDI_NS}}}BPMNShape', shape_attrib)
        bounds = ET.SubElement(shape, f'{{{OMGDC_NS}}}Bounds', {'x': str(x + (i * 220)), 'y': str(y), 'width': str(dx), 'height': str(dy)})

    # For each sequenceFlow create an edge element
    for idx, flow in enumerate(flows, start=1):
        source = flow.get('source')
        target = flow.get('target')
        if source not in created_ids or target not in created_ids:
            continue
        edge_attrib = {'id': f'{process_id}_flow_{idx}_di', 'bpmnElement': f'{process_id}_flow_{idx}'}
        edge = ET.SubElement(plane, f'{{{BPMNDI_NS}}}BPMNEdge', edge_attrib)
        # minimal waypoint points
        ET.SubElement(edge, f'{{{OMGDI_NS}}}waypoint', {'x': '0', 'y': '0'})
        ET.SubElement(edge, f'{{{OMGDI_NS}}}waypoint', {'x': '10', 'y': '10'})

    # Pretty print
    xml = prettify_xml(definitions)
    return xml


# Example quick test when run as script
if __name__ == '__main__':
    sample = {
      "bpmn": {
        "nodes": [
          {"id":"start_1","type":"StartEvent","name":"Start Process","lane":"Requester"},
          {"id":"task_1","type":"UserTask","name":"Submit Request","lane":"Requester"},
          {"id":"gateway_1","type":"ExclusiveGateway","name":"Request Approved?","lane":"Manager"},
          {"id":"task_2","type":"ManualTask","name":"Review Request","lane":"Manager"},
          {"id":"task_3","type":"ServiceTask","name":"Notify via Email","lane":"System"},
          {"id":"task_4","type":"Task","name":"Archive Request","lane":"System"},
          {"id":"gateway_2","type":"ParallelGateway","name":"Parallel Processing","lane":"System"},
          {"id":"data_1","type":"DataObject","name":"Request Form","lane":"Requester"},
          {"id":"end_1","type":"EndEvent","name":"Process Completed","lane":"System"}
        ],
        "flows": [
          {"source":"start_1","target":"task_1","type":"SequenceFlow"},
          {"source":"task_1","target":"data_1","type":"DataAssociation"},
          {"source":"task_1","target":"gateway_1","type":"SequenceFlow"},
          {"source":"gateway_1","target":"task_2","type":"SequenceFlow","condition":"approved == true"},
          {"source":"gateway_1","target":"end_1","type":"SequenceFlow","condition":"approved == false"},
          {"source":"task_2","target":"gateway_2","type":"SequenceFlow"},
          {"source":"gateway_2","target":"task_3","type":"SequenceFlow"},
          {"source":"gateway_2","target":"task_4","type":"SequenceFlow"},
          {"source":"task_3","target":"end_1","type":"MessageFlow"},
          {"source":"task_4","target":"end_1","type":"SequenceFlow"}
        ]
      }
    }

    xml_output = generate_bpmn_xml(sample, process_id='p1', process_name='Sample Process')
    with open('output_bpmn/sample_process.bpmn', 'w', encoding='utf-8') as f:
        f.write(xml_output)
    print("BPMN XML saved to output_bpmn/sample_process.bpmn")
