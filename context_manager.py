import pandas as pd

class ConversationalSession:
    """
    Manages stateful multi-turn conversational execution state matrix.
    Enables sequential query chaining where each operation acts on the preceding state.
    """
    def __init__(self, base_df: pd.DataFrame):
        self.base_df = base_df.copy()
        self.state_stack = [base_df.copy()]
        self.history = []

    @property
    def current_df(self) -> pd.DataFrame:
        return self.state_stack[-1]

    def push_state(self, new_df: pd.DataFrame, query: str, intent: str, entities: dict, code_snippet: str, status_msg: str):
        """
        Pushes a new transformed DataFrame state onto the execution stack.
        """
        self.state_stack.append(new_df.copy())
        turn_number = len(self.history) + 1
        self.history.append({
            "turn": turn_number,
            "query": query,
            "intent": intent,
            "entities": entities,
            "code_snippet": code_snippet,
            "status_message": status_msg,
            "row_count": len(new_df)
        })

    def undo_state(self) -> bool:
        """
        Rolls back the last executed query turn. Returns True if successfully undone.
        """
        if len(self.state_stack) > 1:
            self.state_stack.pop()
            if self.history:
                self.history.pop()
            return True
        return False

    def reset_session(self):
        """
        Resets the conversational session back to the base dataset state.
        """
        self.state_stack = [self.base_df.copy()]
        self.history = []
