import pytest
import pandas as pd
import numpy as np
import httpx

from cleaner import clean_and_profile_dataframe, detect_anomalies, impute_missing_values, encode_categorical_features, generate_data_profile
from nlp_engine import AdvancedNLPEngine
from executor import execute_nlp_query
from narrative import generate_narrative_summary
from benchmark import run_benchmark_suite
from version_manager import VersionManager
from duckdb_engine import FastDuckDBEngine
from agent_planner import AgentPlanner
from main_api import app

@pytest.fixture
def anyio_backend():
    return 'asyncio'

@pytest.fixture
def sample_df():
    return pd.DataFrame({
        'Department': ['Engineering', 'Enginering', 'Marketing', 'Marketng', 'Sales'],
        'Salary': [75000, 82000, 68000, 71000, 950000],
        'Age': [29, 51, np.nan, 45, 26],
        'Name': ['John Doe', 'Jane Smith', 'Bob Johnson', 'Alice Brown', 'Charlie White']
    })

def test_cleaner_fuzzy_typo_and_duplicates(sample_df):
    cleaned = clean_and_profile_dataframe(sample_df, score_cutoff=80)
    unique_depts = cleaned['Department'].unique()
    assert 'Engineering' in unique_depts
    assert 'Enginering' not in unique_depts

def test_version_manager_commit_and_rollback(sample_df):
    vm = VersionManager(sample_df)
    assert len(vm.history) == 1
    assert vm.current_index == 0
    
    modified = sample_df.copy()
    modified.loc[0, 'Salary'] = 90000
    
    draft, diff_sum = vm.create_draft(modified, "Update salary of John", "UPDATE_VALUE", "df.loc[0, 'Salary'] = 90000")
    assert diff_sum["modified_cells"] == 1
    
    vm.confirm_draft()
    assert len(vm.history) == 2
    assert vm.get_active_df().loc[0, 'Salary'] == 90000
    
    vm.rollback_to(0)
    assert vm.get_active_df().loc[0, 'Salary'] == 75000

def test_fast_duckdb_engine_sql(sample_df):
    cleaned = clean_and_profile_dataframe(sample_df)
    d_engine = FastDuckDBEngine()
    query = "SELECT Department, AVG(Salary) AS avg_sal FROM dataset GROUP BY Department ORDER BY avg_sal DESC"
    res = d_engine.execute_sql_query(cleaned, query)
    assert not res.empty
    assert 'avg_sal' in res.columns
    assert len(res) == 3

def test_agent_planner_decomposition():
    planner = AgentPlanner()
    steps = planner.decompose_query("In age column where age > 40 replace it with 20 and then sort by salary descending")
    assert len(steps) == 2
    assert steps[0]["query"] == "In age column where age > 40 replace it with 20"
    assert steps[1]["query"] == "sort by salary descending"

def test_retrieval_query_for_person_name(sample_df):
    engine = AdvancedNLPEngine()
    query = "NEED THE DATA OF JOHN NAME"
    intent = engine.predict_intent(query)
    assert intent == "FILTER"
    
    entities = engine.extract_entities(query, sample_df.columns.tolist())
    assert 'Name' in entities['columns']
    
    res = execute_nlp_query(sample_df, intent, entities)
    assert res['status'] == 'success'
    filtered_df = res['transformed_df']
    assert len(filtered_df) == 1
    assert filtered_df.iloc[0]['Name'] == 'John Doe'

def test_deep_semantic_vector_intents():
    engine = AdvancedNLPEngine()
    assert engine.predict_intent("Can you calculate the total sum of sales?") == "SUM"
    assert engine.predict_intent("Give me a breakdown of average earnings for each team") in ["GROUP_BY", "AVERAGE"]
    assert engine.predict_intent("Display all employees with missing age") == "SHOW_NULLS"
    assert engine.predict_intent("Find statistical anomalies in price") == "ANOMALY"

def test_word2number_resolution():
    engine = AdvancedNLPEngine()
    nums = engine.parse_numbers_and_words("where age is above forty replace it with twenty")
    assert 40 in nums
    assert 20 in nums

def test_semantic_column_synonyms():
    engine = AdvancedNLPEngine()
    cols = ['Department', 'Salary', 'Age', 'Name']
    matched = engine.resolve_semantic_column(['earnings', 'team', 'staff'], cols)
    assert 'Salary' in matched
    assert 'Department' in matched
    assert 'Name' in matched

def test_universal_semantic_clause_replacement(sample_df):
    engine = AdvancedNLPEngine()
    query = "IN AGE COLUMN WHERE AGE IS ABOVE 40 REPLACE IT WITH 20"
    intent = engine.predict_intent(query)
    assert intent == "UPDATE_VALUE"
    
    entities = engine.extract_entities(query, sample_df.columns.tolist())
    assert 'Age' in entities['columns']
    assert entities['operator'] == '>'
    
    res = execute_nlp_query(sample_df, intent, entities)
    assert res['status'] == 'success'
    updated_df = res['transformed_df']
    assert (updated_df['Age'] > 40).sum() == 0

def test_mice_imputation(sample_df):
    imputed = impute_missing_values(sample_df, method='mice')
    assert imputed['Age'].isnull().sum() == 0

def test_feature_encoding(sample_df):
    encoded = encode_categorical_features(sample_df, method='onehot')
    assert 'Department_Marketing' in encoded.columns or 'Department_Sales' in encoded.columns

def test_narrative_summary_generation(sample_df):
    narrative = generate_narrative_summary(sample_df)
    assert "data_hygiene_score" in narrative
    assert len(narrative["key_insights"]) > 0

def test_median_mode_and_add_operations(sample_df):
    engine = AdvancedNLPEngine()
    
    # Test Median
    med_intent = engine.predict_intent("calculate median salary")
    assert med_intent == "MEDIAN"
    med_res = execute_nlp_query(sample_df, med_intent, {"columns": ["Salary"], "raw_query": "calculate median salary"})
    assert med_res["result_metric"]["value"] == 75000.0

    # Test Mode
    mode_intent = engine.predict_intent("find mode of department")
    assert mode_intent == "MODE"
    mode_res = execute_nlp_query(sample_df, mode_intent, {"columns": ["Department"], "raw_query": "find mode of department"})
    assert mode_res["result_metric"]["value"] is not None

    # Test Add Column
    add_col_intent = engine.predict_intent("add column Bonus with value 5000")
    assert add_col_intent == "ADD_COL"
    add_col_res = execute_nlp_query(sample_df, add_col_intent, {"columns": [], "numbers": [5000], "literals": [], "raw_query": "add column Bonus with value 5000"})
    assert "Bonus" in add_col_res["transformed_df"].columns

    # Test Add Row
    add_row_intent = engine.predict_intent("add row with name Alex salary 60000")
    assert add_row_intent == "ADD_ROW"
    add_row_res = execute_nlp_query(sample_df, add_row_intent, {"columns": [], "numbers": [60000], "literals": ["Alex"], "raw_query": "add row with name Alex salary 60000"})
    assert len(add_row_res["transformed_df"]) == len(sample_df) + 1

@pytest.mark.anyio
async def test_fastapi_endpoints():
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        res_health = await client.get("/health")
        assert res_health.status_code == 200

if __name__ == "__main__":
    pytest.main(["-v", "test_suite.py"])

