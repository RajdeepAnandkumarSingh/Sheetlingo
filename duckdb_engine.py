import duckdb
import pandas as pd
from typing import Dict, Any

class FastDuckDBEngine:
    """
    High-Performance In-Memory SQL Engine using DuckDB.
    Provides zero-copy SQL execution directly on Pandas DataFrames, fast aggregation,
    and multi-sheet relational joins (VLOOKUP replacement).
    """
    def __init__(self):
        self.con = duckdb.connect(database=':memory:')

    def execute_sql_query(self, df: pd.DataFrame, sql_query: str) -> pd.DataFrame:
        """
        Executes an in-memory SQL query directly against a Pandas DataFrame.
        Refer to the DataFrame inside the SQL string as 'dataset'.
        Example: SELECT * FROM dataset WHERE Age > 40 ORDER BY Salary DESC
        """
        if df is None or df.empty:
            return pd.DataFrame()
            
        self.con.register('dataset', df)
        try:
            result_df = self.con.execute(sql_query).df()
        finally:
            self.con.unregister('dataset')
        return result_df

    def fast_aggregate(self, df: pd.DataFrame, group_col: str, agg_col: str, agg_func: str = "AVG") -> pd.DataFrame:
        """
        Computes high-speed grouped aggregation using DuckDB SQL vectorization.
        """
        valid_funcs = {"AVG": "AVG", "SUM": "SUM", "COUNT": "COUNT", "MIN": "MIN", "MAX": "MAX"}
        func = valid_funcs.get(agg_func.upper(), "AVG")
        
        query = f"""
            SELECT "{group_col}", {func}("{agg_col}") AS {func.lower()}_{agg_col}
            FROM dataset
            GROUP BY "{group_col}"
            ORDER BY {func.lower()}_{agg_col} DESC
        """
        return self.execute_sql_query(df, query)

    def fuzzy_relational_join(self, df_left: pd.DataFrame, df_right: pd.DataFrame, left_key: str, right_key: str, join_type: str = "INNER") -> pd.DataFrame:
        """
        Merges two DataFrames (multi-sheet join) based on key columns using DuckDB SQL.
        """
        if df_left is None or df_right is None:
            return pd.DataFrame()
            
        self.con.register('table_a', df_left)
        self.con.register('table_b', df_right)
        
        j_type = join_type.upper() if join_type.upper() in ["INNER", "LEFT", "RIGHT", "FULL"] else "INNER"
        
        query = f"""
            SELECT table_a.*, table_b.* EXCLUDE ("{right_key}")
            FROM table_a
            {j_type} JOIN table_b ON table_a."{left_key}" = table_b."{right_key}"
        """
        try:
            result_df = self.con.execute(query).df()
        finally:
            self.con.unregister('table_a')
            self.con.unregister('table_b')
            
        return result_df
