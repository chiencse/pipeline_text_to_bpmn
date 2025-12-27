"""
Test script to debug interrupt behavior in scenario_b pipeline
"""
import os
os.environ["LANGGRAPH_USE_CHECKPOINTER"] = "true"  # Force checkpointer on

from app.scenario_b import build_graph_b
import uuid

# Build graph with checkpointer
graph = build_graph_b()

# Create test input
thread_id = str(uuid.uuid4())
config = {"configurable": {"thread_id": thread_id}}
initial_state = {
    "text": "Send an email to finance, attach the quotation, wait for reply, then update the invoice in the SAP system."
}

print(f"=" * 80)
print(f"Starting pipeline with thread_id: {thread_id}")
print(f"=" * 80)

# Stream through events
event_count = 0
try:
    for event in graph.stream(initial_state, config, stream_mode="updates"):
        event_count += 1
        node_names = list(event.keys())
        print(f"\n[Event {event_count}] Nodes: {node_names}")
        
        # Print abbreviated state for each node
        for node_name, node_state in event.items():
            if isinstance(node_state, dict):
                keys = list(node_state.keys())
                print(f"  - {node_name}: keys = {keys[:5]}{'...' if len(keys) > 5 else ''}")
                
                # Check for interrupt markers
                if "__pending_interrupt__" in node_state:
                    print(f"    🔴 INTERRUPT MARKER FOUND!")
                    print(f"    Type: {node_state['__pending_interrupt__'].get('type')}")
                
except Exception as e:
    print(f"\n❌ Exception during stream: {type(e).__name__}: {str(e)}")

print(f"\n{' ' * 80}")
print(f"Stream completed with {event_count} events")
print(f"=" * 80)

# Check checkpoint state
checkpoint_state = graph.get_state(config)

print(f"\n📊 Checkpoint State:")
print(f"  - Has checkpoint: {checkpoint_state is not None}")

if checkpoint_state:
    print(f"  - Next nodes: {checkpoint_state.next}")
    print(f"  - Config: {checkpoint_state.config}")
    
    if checkpoint_state.values:
        values = checkpoint_state.values
        print(f"\n📦 State Values:")
        print(f"  - Keys: {list(values.keys())}")
        
        # Check for interrupt markers
        has_interrupt = "__pending_interrupt__" in values
        print(f"  - Has __pending_interrupt__: {has_interrupt}")
        
        if has_interrupt:
            payload = values["__pending_interrupt__"]
            print(f"\n🎯 Interrupt Payload:")
            print(f"  - Type: {payload.get('type')}")
            print(f"  - Instruction: {payload.get('instruction', 'N/A')}")
            print(f"  - Has BPMN: {'bpmn' in payload}")
        else:
            print(f"\n⚠️  No interrupt marker found in state!")
            print(f"  - user_decision: {values.get('user_decision', 'NOT SET')}")
            print(f"  - Has BPMN: {'bpmn' in values}")
    else:
        print(f"  - Values: None")
else:
    print(f"  ❌ No checkpoint state found!")

print(f"\n{'=' * 80}")
print(f"Test completed. Thread ID: {thread_id}")
print(f"{'=' * 80}")

