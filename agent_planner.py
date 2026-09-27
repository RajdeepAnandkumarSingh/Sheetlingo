import re
from typing import List, Dict, Any

class AgentPlanner:
    """
    Multi-Step Agentic Command Deconstruction & Planning Pipeline.
    Breaks complex compound natural language prompts into sequential atomic tasks.
    """
    def __init__(self):
        # Conjunction split regex matching 'then', 'and then', 'after that', 'and next', 'followed by', or semicolons/commas
        self.split_pattern = re.compile(
            r'\b(?:then|and then|after that|and next|followed by|afterwards|also)\b|;', 
            re.IGNORECASE
        )

    def decompose_query(self, full_query: str) -> List[Dict[str, Any]]:
        """
        Deconstructs a compound multi-step query into an ordered list of execution steps.
        Example: 'In age column where age > 40 replace it with 20 and then sort by salary descending'
        -> [
             {"step": 1, "query": "In age column where age > 40 replace it with 20"},
             {"step": 2, "query": "sort by salary descending"}
           ]
        """
        if not full_query or not full_query.strip():
            return [{"step": 1, "query": ""}]

        raw_parts = self.split_pattern.split(full_query)
        clean_steps = []
        step_counter = 1

        for part in raw_parts:
            part_str = part.strip()
            if len(part_str) >= 3:
                clean_steps.append({
                    "step": step_counter,
                    "query": part_str
                })
                step_counter += 1

        if not clean_steps:
            clean_steps = [{"step": 1, "query": full_query.strip()}]

        return clean_steps
