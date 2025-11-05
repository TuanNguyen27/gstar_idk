"""
Prompt Templates for Router Training
Defines the instruction format and reasoning chains.
"""

from typing import Dict
from config import POLICIES

def get_routes_xml() -> str:
    """Generate the <routes> XML section for the router prompt."""
    routes_xml = "<routes>\n"
    for policy_name, policy_info in POLICIES.items():
        routes_xml += f"  <route>\n"
        routes_xml += f"    <name>{policy_name}</name>\n"
        routes_xml += f"    <description>{policy_info['description']}</description>\n"
        routes_xml += f"  </route>\n"
    routes_xml += "</routes>"
    return routes_xml

def create_router_prompt(query: str) -> str:
    """
    Create the full router instruction prompt.

    Args:
        query: User query to route

    Returns:
        Formatted instruction prompt
    """
    routes_xml = get_routes_xml()

    prompt = f"""[INST]
You are an expert query routing assistant. Your task is to analyze the user's query and select the most appropriate policy from the list below.

{routes_xml}

<conversation>
{query}
</conversation>

Analyze the query and respond *only* with your reasoning and final decision in the specified JSON format.
[/INST]"""

    return prompt

def create_reasoning_chain(query_analysis: str, policy: str) -> str:
    """
    Create a reasoning chain (CoT) for a given policy decision.

    Args:
        query_analysis: One-sentence analysis of the query
        policy: Policy name (Standard_Query, Complex_Query, Ambiguous_Query)

    Returns:
        Formatted reasoning chain with [REASONING] and [DECISION] markers
    """
    reasoning_templates = {
        "Standard_Query": f"[REASONING] Query Analysis: '{query_analysis}'. This type of query is simple enough for the standard model. [DECISION] Standard_Query",
        "Complex_Query": f"[REASONING] Query Analysis: '{query_analysis}'. This type of query is complex and requires escalation for an accurate response. [DECISION] Complex_Query",
        "Ambiguous_Query": f"[REASONING] Query Analysis: '{query_analysis}'. This type of query is ambiguous and unlikely to be answered well by any model. Defaulting to the efficient route. [DECISION] Ambiguous_Query",
    }

    if policy not in reasoning_templates:
        raise ValueError(f"Unknown policy: {policy}")

    return reasoning_templates[policy]

def create_rejected_reasoning(query_analysis: str, correct_policy: str) -> str:
    """
    Create a rejected (incorrect) reasoning chain for DPO training.

    Args:
        query_analysis: One-sentence analysis of the query
        correct_policy: The correct policy

    Returns:
        Reasoning chain for an incorrect policy
    """
    # Map correct policy to a wrong policy
    wrong_policy_map = {
        "Standard_Query": "Complex_Query",  # Over-escalate
        "Complex_Query": "Standard_Query",  # Under-escalate
        "Ambiguous_Query": "Standard_Query",  # Ignore ambiguity
    }

    wrong_policy = wrong_policy_map[correct_policy]
    return create_reasoning_chain(query_analysis, wrong_policy)
