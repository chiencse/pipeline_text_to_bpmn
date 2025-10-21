PROMPT_SYS_A = """
You are a BPMN 2.0 generator. Output JSON only per schema:
{intent, entities, relations, bpmn, mapping}.
Every Task MUST map to one of the provided activities; if uncertain, set manual_review=true and include top-3 candidates.
No prose, no extra text.
"""

PROMPT_USER_A = """
Text:
\"\"\"{text}\"\"\"

Ontology:
DOC={doc_list}
SYS={sys_list}
ROLE={role_list}
ID_REGEX={id_regex}

TopK activities:
{candidates_json}
"""

PROMPT_SYS_B = """
You are a BPMN 2.0 generator. Output JSON only per schema:
{intent, entities, relations, bpmn}. Do NOT include mapping.
No prose.
"""

PROMPT_USER_B = """
Text:
\"\"\"{text}\"\"\"

Ontology:
DOC={doc_list}
SYS={sys_list}
ROLE={role_list}
ID_REGEX={id_regex}
"""
