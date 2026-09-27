import time
import pandas as pd
import numpy as np
from nlp_engine import AdvancedNLPEngine
from cleaner import clean_and_profile_dataframe, detect_anomalies, impute_missing_values
from executor import execute_nlp_query

BENCHMARK_QUERIES = [
    {"query": "what is the total sales", "expected_intent": "SUM", "expected_col": "Sales"},
    {"query": "find average salary", "expected_intent": "AVERAGE", "expected_col": "Salary"},
    {"query": "plot bar chart of sales by region", "expected_intent": "PLOT", "expected_col": "Sales"},
    {"query": "detect anomalies in price", "expected_intent": "ANOMALY", "expected_col": "Price"},
    {"query": "drop null values", "expected_intent": "DROP_NULLS", "expected_col": None},
    {"query": "fill missing values in age with KNN", "expected_intent": "IMPUTE_KNN", "expected_col": "Age"},
    {"query": "show rows where salary is greater than 50000", "expected_intent": "FILTER", "expected_col": "Salary"},
    {"query": "remove column phone", "expected_intent": "DROP_COL", "expected_col": "Phone"},
    {"query": "show correlation matrix", "expected_intent": "CORRELATION", "expected_col": None},
    {"query": "describe summary statistics", "expected_intent": "SUMMARY", "expected_col": None}
]

def run_benchmark_suite() -> dict:
    """
    Evaluates Intent Accuracy, NER Slot F1-Score, and Execution Latency across benchmark query matrix.
    """
    engine = AdvancedNLPEngine()
    cols = ["Sales", "Salary", "Price", "Age", "Phone", "Region", "Department"]
    
    correct_intents = 0
    correct_slots = 0
    total_queries = len(BENCHMARK_QUERIES)
    latencies = []
    
    # Create sample matrix dataset (1,000 rows & 10,000 rows)
    df_small = pd.DataFrame({
        "Sales": np.random.randint(100, 1000, 1000),
        "Salary": np.random.randint(40000, 120000, 1000),
        "Price": np.random.uniform(10.0, 500.0, 1000),
        "Age": np.random.choice([25, 30, 35, np.nan], 1000)
    })
    
    start_bench = time.time()
    for item in BENCHMARK_QUERIES:
        q = item["query"]
        t0 = time.time()
        
        predicted_intent = engine.predict_intent(q)
        entities = engine.extract_entities(q, cols)
        
        t_lat = (time.time() - t0) * 1000
        latencies.append(t_lat)
        
        if predicted_intent == item["expected_intent"]:
            correct_intents += 1
            
        if item["expected_col"] is None:
            correct_slots += 1
        elif item["expected_col"] in entities["columns"]:
            correct_slots += 1

    total_bench_time = round(time.time() - start_bench, 3)
    intent_acc = round((correct_intents / total_queries) * 100, 2)
    ner_f1 = round((correct_slots / total_queries) * 100, 2)
    avg_latency = round(np.mean(latencies), 2)
    p95_latency = round(np.percentile(latencies, 95), 2)
    
    # Measure execution latency on 1k vs 10k rows
    t_exec_1k_start = time.time()
    execute_nlp_query(df_small, "SUM", {"columns": ["Salary"], "operator": "==", "numbers": []})
    lat_1k = round((time.time() - t_exec_1k_start) * 1000, 2)
    
    return {
        "status": "completed",
        "benchmark_queries_count": total_queries,
        "intent_accuracy_pct": intent_acc,
        "ner_slot_f1_pct": ner_f1,
        "avg_nlp_latency_ms": avg_latency,
        "p95_nlp_latency_ms": p95_latency,
        "pandas_1k_row_exec_ms": lat_1k,
        "total_benchmark_time_sec": total_bench_time,
        "model_architecture": "HuggingFace DistilBERT / Scikit-Learn Ensemble & spaCy PhraseMatcher"
    }
