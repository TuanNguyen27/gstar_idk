"""
Oracle Logic: 9-Cell Matrix for Policy Labeling
Maps (medium_label, large_label) → Final_Policy_Label
"""

from typing import Tuple
from .models import PolicyType, JudgmentLabel

# The 9-Cell Oracle Matrix
ORACLE_MATRIX: dict[Tuple[JudgmentLabel, JudgmentLabel], PolicyType] = {
    # Medium Correct
    ("Correct", "Correct"): "Standard_Query",    # Both correct → optimize cost
    ("Correct", "Incorrect"): "Standard_Query",  # Medium correct → use medium
    ("Correct", "IDK"): "Standard_Query",        # Medium correct is best

    # Medium Incorrect
    ("Incorrect", "Correct"): "Complex_Query",   # Classic escalation
    ("Incorrect", "Incorrect"): "Ambiguous_Query",  # Both failed → don't escalate
    ("Incorrect", "IDK"): "Complex_Query",       # IDK safer than incorrect

    # Medium IDK
    ("IDK", "Correct"): "Complex_Query",         # Large knows answer
    ("IDK", "Incorrect"): "Ambiguous_Query",     # IDK was safer
    ("IDK", "IDK"): "Ambiguous_Query",          # Both failed → don't escalate
}

def get_policy_label(medium_label: JudgmentLabel, large_label: JudgmentLabel) -> PolicyType:
    """
    Apply oracle matrix logic to determine final policy label.

    Args:
        medium_label: Judgment for medium model's answer
        large_label: Judgment for large model's answer

    Returns:
        Final policy label for training
    """
    return ORACLE_MATRIX[(medium_label, large_label)]

def get_matrix_rationale(medium_label: JudgmentLabel, large_label: JudgmentLabel) -> str:
    """Get human-readable rationale for a matrix decision."""
    rationales = {
        ("Correct", "Correct"): "Both correct. Optimize for cost.",
        ("Correct", "Incorrect"): "Medium is correct. Optimize for cost/correctness.",
        ("Correct", "IDK"): "Medium is correct, which is the best outcome.",
        ("Incorrect", "Correct"): "Classic escalation. Large model is required.",
        ("Incorrect", "Incorrect"): "Both failed. Don't escalate; save cost.",
        ("Incorrect", "IDK"): "Escalate. IDK is safer than Incorrect.",
        ("IDK", "Correct"): "Classic escalation. Large model knows the answer.",
        ("IDK", "Incorrect"): "Medium was safer (IDK). Don't escalate.",
        ("IDK", "IDK"): "Both failed. Don't escalate; save cost.",
    }
    return rationales[(medium_label, large_label)]
