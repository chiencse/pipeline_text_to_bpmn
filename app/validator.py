def validate_bpmn(bpmn: dict, scenario: str) -> list[str]:
    errors = []
    nodes = {n["id"]: n for n in bpmn.get("nodes", [])}
    flows = bpmn.get("flows", [])
    # 1 Start + 1 End
    if sum(1 for n in nodes.values() if n["type"] == "StartEvent") != 1:
        errors.append("Must have exactly one StartEvent.")
    if sum(1 for n in nodes.values() if n["type"] == "EndEvent") != 1:
        errors.append("Must have exactly one EndEvent.")
    # Valid flow endpoints
    for f in flows:
        if f["source"] not in nodes or f["target"] not in nodes:
            errors.append(f"Invalid flow endpoint: {f}")
    # Mapping policy
    if scenario == "A":
        # Scenario A yêu cầu map (trừ khi cho phép AdapterTask explicitly)
        pass
    return errors
