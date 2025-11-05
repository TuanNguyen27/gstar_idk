"""
Flexible Policy Mapping
Allows dynamic, query-dependent model selection beyond static JSON mapping.
"""

from typing import Dict, Callable, Optional
import re

class FlexiblePolicyMapper:
    """
    Maps policies to models with query-dependent logic.

    This addresses the concern that static policy_map.json is too rigid.
    You can add custom rules based on query characteristics, time, cost budgets, etc.
    """

    def __init__(
        self,
        default_map: Dict[str, str],
        custom_rules: Optional[Dict[str, Callable]] = None,
    ):
        """
        Initialize flexible policy mapper.

        Args:
            default_map: Default policy → model mapping (fallback)
            custom_rules: Dict of custom rule functions {rule_name: rule_func}
                         Each rule_func takes (query, policy) → Optional[model_name]
        """
        self.default_map = default_map
        self.custom_rules = custom_rules or {}

    def get_model(self, query: str, policy: str) -> str:
        """
        Map policy to model with query-dependent logic.

        Args:
            query: User query
            policy: Routed policy from router

        Returns:
            Target model name
        """
        # Apply custom rules in order
        for rule_name, rule_func in self.custom_rules.items():
            model = rule_func(query, policy)
            if model is not None:
                return model

        # Fallback to default mapping
        return self.default_map.get(policy, "gemini-2.5-flash")

    @staticmethod
    def create_cost_aware_mapper(
        default_map: Dict[str, str],
        cost_budget: float = None,
        current_spend: float = 0.0,
    ) -> "FlexiblePolicyMapper":
        """
        Create a cost-aware mapper that downgrades to cheaper models near budget limit.

        Args:
            default_map: Default mapping
            cost_budget: Maximum cost budget
            current_spend: Current spend so far

        Returns:
            FlexiblePolicyMapper instance
        """
        def cost_rule(query: str, policy: str) -> Optional[str]:
            if cost_budget is None:
                return None

            # If near budget (>90%), always use Flash
            if current_spend >= cost_budget * 0.9:
                return "gemini-2.5-flash"

            return None

        return FlexiblePolicyMapper(
            default_map=default_map,
            custom_rules={"cost_budget": cost_rule},
        )

    @staticmethod
    def create_keyword_based_mapper(
        default_map: Dict[str, str],
        keyword_overrides: Dict[str, str],
    ) -> "FlexiblePolicyMapper":
        """
        Create a mapper that overrides based on query keywords.

        Example:
            keyword_overrides = {
                "code": "gemini-2.5-pro",  # Always use Pro for code
                "math": "gemini-2.5-pro",  # Always use Pro for math
            }

        Args:
            default_map: Default mapping
            keyword_overrides: Dict of keyword → forced model

        Returns:
            FlexiblePolicyMapper instance
        """
        def keyword_rule(query: str, policy: str) -> Optional[str]:
            query_lower = query.lower()
            for keyword, model in keyword_overrides.items():
                if keyword in query_lower:
                    return model
            return None

        return FlexiblePolicyMapper(
            default_map=default_map,
            custom_rules={"keyword_override": keyword_rule},
        )

    @staticmethod
    def create_hybrid_mapper(
        default_map: Dict[str, str],
        cost_budget: float = None,
        current_spend: float = 0.0,
        keyword_overrides: Dict[str, str] = None,
        length_threshold: int = 500,
    ) -> "FlexiblePolicyMapper":
        """
        Create a hybrid mapper with multiple rules.

        Args:
            default_map: Default mapping
            cost_budget: Cost budget for budget rule
            current_spend: Current spend
            keyword_overrides: Keyword → model overrides
            length_threshold: Query length threshold for complexity estimation

        Returns:
            FlexiblePolicyMapper instance
        """
        rules = {}

        # Rule 1: Budget constraint
        if cost_budget is not None:
            def cost_rule(query: str, policy: str) -> Optional[str]:
                if current_spend >= cost_budget * 0.9:
                    return "gemini-2.5-flash"
                return None
            rules["cost_budget"] = cost_rule

        # Rule 2: Keyword override
        if keyword_overrides:
            def keyword_rule(query: str, policy: str) -> Optional[str]:
                query_lower = query.lower()
                for keyword, model in keyword_overrides.items():
                    if keyword in query_lower:
                        return model
                return None
            rules["keyword_override"] = keyword_rule

        # Rule 3: Length-based heuristic
        def length_rule(query: str, policy: str) -> Optional[str]:
            # Very long queries → likely complex
            if len(query) > length_threshold and policy == "Standard_Query":
                return "gemini-2.5-pro"  # Upgrade
            return None
        rules["length_heuristic"] = length_rule

        # Rule 4: Math detection (upgrade Standard to Pro for math)
        def math_rule(query: str, policy: str) -> Optional[str]:
            math_indicators = [
                r'\d+\s*[\+\-\*\/\^]\s*\d+',  # Arithmetic
                r'equation',
                r'calculate',
                r'derivative',
                r'integral',
            ]
            if policy == "Standard_Query":
                for pattern in math_indicators:
                    if re.search(pattern, query.lower()):
                        return "gemini-2.5-pro"
            return None
        rules["math_detection"] = math_rule

        return FlexiblePolicyMapper(
            default_map=default_map,
            custom_rules=rules,
        )


# Example usage configurations

# 1. Static mapping (original approach)
STATIC_MAP = {
    "Standard_Query": "gemini-2.5-flash",
    "Complex_Query": "gemini-2.5-pro",
    "Ambiguous_Query": "gemini-2.5-flash",
}

# 2. Cost-aware mapping
def get_cost_aware_mapper(budget: float = 10.0, spend: float = 0.0):
    return FlexiblePolicyMapper.create_cost_aware_mapper(
        default_map=STATIC_MAP,
        cost_budget=budget,
        current_spend=spend,
    )

# 3. Keyword-based mapping
def get_keyword_mapper():
    return FlexiblePolicyMapper.create_keyword_based_mapper(
        default_map=STATIC_MAP,
        keyword_overrides={
            "code": "gemini-2.5-pro",
            "programming": "gemini-2.5-pro",
            "math": "gemini-2.5-pro",
            "calculate": "gemini-2.5-pro",
        }
    )

# 4. Hybrid mapping (recommended)
def get_hybrid_mapper(budget: float = None, spend: float = 0.0):
    return FlexiblePolicyMapper.create_hybrid_mapper(
        default_map=STATIC_MAP,
        cost_budget=budget,
        current_spend=spend,
        keyword_overrides={
            "code": "gemini-2.5-pro",
            "programming": "gemini-2.5-pro",
        },
        length_threshold=500,
    )
