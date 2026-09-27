import time
import hashlib
import pandas as pd
from typing import Dict, List, Tuple

class DatasetCommit:
    """
    Represents an immutable version snapshot of the dataset with Git-style metadata.
    """
    def __init__(self, commit_id: str, df: pd.DataFrame, message: str, intent: str, code_trace: str):
        self.commit_id = commit_id
        self.timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        self.df = df.copy()
        self.message = message
        self.intent = intent
        self.code_trace = code_trace
        self.row_count = len(df)
        self.col_count = len(df.columns)

class VersionManager:
    """
    Git-Style Version Control Engine for Spreadsheets.
    Supports draft diff previews, cell modification counts, explicit commits, and 1-click rollbacks.
    """
    def __init__(self, initial_df: pd.DataFrame):
        self.history: List[DatasetCommit] = []
        self.current_index = -1
        self.draft_df: pd.DataFrame = None
        self.pending_message = ""
        self.pending_intent = ""
        self.pending_trace = ""
        self.commit("v1_initial", initial_df, "Initial Dataset Ingestion & Sanitization", "INGEST", "# Raw Data Ingested")

    def commit(self, version_tag: str, df: pd.DataFrame, message: str, intent: str, code_trace: str):
        """
        Commits a new dataset snapshot to the historical version stack.
        Truncates any forward history if committing after a rollback.
        """
        if self.current_index < len(self.history) - 1:
            self.history = self.history[:self.current_index + 1]
            
        commit_obj = DatasetCommit(version_tag, df, message, intent, code_trace)
        self.history.append(commit_obj)
        self.current_index += 1
        self.draft_df = None
        self.pending_message = ""
        self.pending_intent = ""
        self.pending_trace = ""

    def create_draft(self, modified_df: pd.DataFrame, message: str, intent: str, code_trace: str) -> Tuple[pd.DataFrame, Dict]:
        """
        Calculates exact cell-by-cell diffs between current active state and proposed state without mutating current state.
        """
        self.draft_df = modified_df.copy()
        self.pending_message = message
        self.pending_intent = intent
        self.pending_trace = code_trace
        current_df = self.get_active_df()
        
        rows_added = max(0, len(modified_df) - len(current_df))
        rows_removed = max(0, len(current_df) - len(modified_df))
        
        modified_cells = 0
        common_cols = list(set(current_df.columns).intersection(set(modified_df.columns)))
        min_rows = min(len(current_df), len(modified_df))
        
        if min_rows > 0 and len(common_cols) > 0:
            try:
                c_sub = current_df.iloc[:min_rows][common_cols]
                m_sub = modified_df.iloc[:min_rows][common_cols]
                diff_mask = c_sub.ne(m_sub).fillna(True)
                diff_mask &= ~(c_sub.isna() & m_sub.isna())
                modified_cells = int(diff_mask.sum().sum())
            except Exception:
                modified_cells = len(modified_df)

        diff_summary = {
            "rows_added": rows_added,
            "rows_removed": rows_removed,
            "columns_added": len(set(modified_df.columns) - set(current_df.columns)),
            "columns_removed": len(set(current_df.columns) - set(modified_df.columns)),
            "modified_cells": modified_cells,
            "total_rows_after": len(modified_df),
            "total_cols_after": len(modified_df.columns),
            "active_version": self.get_active_commit().commit_id
        }
        return self.draft_df, diff_summary

    def confirm_draft(self):
        """
        Commits the currently pending draft into the Git version history.
        """
        if self.draft_df is not None:
            v_tag = f"v{len(self.history) + 1}_{self.pending_intent.lower()}"
            self.commit(v_tag, self.draft_df, self.pending_message, self.pending_intent, self.pending_trace)

    def discard_draft(self):
        """
        Discards the pending draft and retains current state.
        """
        self.draft_df = None
        self.pending_message = ""
        self.pending_intent = ""
        self.pending_trace = ""

    def rollback_to(self, index: int) -> pd.DataFrame:
        """
        Rolls back the active state to an arbitrary historical commit index.
        """
        if 0 <= index < len(self.history):
            self.current_index = index
            self.draft_df = None
            return self.get_active_df()
        return self.get_active_df()

    def get_active_df(self) -> pd.DataFrame:
        return self.history[self.current_index].df.copy()

    def get_active_commit(self) -> DatasetCommit:
        return self.history[self.current_index]


with open(__file__, "rb") as _source_file:
    SHEETLINGO_SOURCE_DIGEST = hashlib.sha256(_source_file.read()).hexdigest()
