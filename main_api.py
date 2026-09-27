import time
import io
import pandas as pd
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from cleaner import clean_and_profile_dataframe, detect_anomalies, generate_data_profile, impute_missing_values, encode_categorical_features
from nlp_engine import AdvancedNLPEngine
from executor import execute_nlp_query
from narrative import generate_narrative_summary
from benchmark import run_benchmark_suite
from duckdb_engine import FastDuckDBEngine
from agent_planner import AgentPlanner

app = FastAPI(
    title="SheetLingo Enterprise OS v4.0 REST API",
    description="Decoupled Microservice OS for Tabular Natural Language Processing, DuckDB SQL Acceleration, Agentic Planning, & Benchmark Analysis",
    version="4.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

engine = AdvancedNLPEngine()
duckdb_engine = FastDuckDBEngine()
agent_planner = AgentPlanner()

def parse_uploaded_file(file: UploadFile, contents: bytes) -> pd.DataFrame:
    try:
        if file.filename.endswith('.csv'):
            df = pd.read_csv(io.BytesIO(contents))
        elif file.filename.endswith(('.xlsx', '.xls')):
            df = pd.read_excel(io.BytesIO(contents))
        else:
            raise HTTPException(status_code=400, detail="Unsupported file format. Please upload a .csv or .xlsx file.")
        return df
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to parse file: {str(e)}")

@app.get("/")
def read_root():
    return {
        "system": "SheetLingo Enterprise OS REST API Engine",
        "status": "online",
        "version": "4.0.0",
        "documentation": "/docs"
    }

@app.get("/health")
def health_check():
    return {"status": "healthy", "nlp_engine": "transformer_deep_semantic_ready", "duckdb": "online"}

@app.post("/process_query/")
async def process_query_endpoint(file: UploadFile = File(...), query: str = Form("")):
    start_time = time.time()
    contents = await file.read()
    df = parse_uploaded_file(file, contents)
    df_cleaned = clean_and_profile_dataframe(df)
    
    intent = engine.predict_intent(query)
    entities = engine.extract_entities(query, df_cleaned.columns.tolist(), dataframe=df_cleaned)
    exec_res = execute_nlp_query(df_cleaned, intent, entities)
    
    execution_time_ms = round((time.time() - start_time) * 1000, 2)
    
    return {
        "status": "success",
        "filename": file.filename,
        "query": query,
        "predicted_intent": intent,
        "extracted_entities": entities,
        "execution_status": exec_res["status_message"],
        "interpretation": exec_res["interpretation"],
        "needs_clarification": exec_res["needs_clarification"],
        "clarification_message": exec_res["clarification_message"],
        "result_metric": exec_res["result_metric"],
        "chart_spec": exec_res["chart_spec"],
        "code_snippet": exec_res["code_snippet"],
        "narrative_summary": exec_res["narrative_summary"],
        "data_profile": generate_data_profile(df_cleaned),
        "execution_latency_ms": execution_time_ms,
        "transformed_row_count": len(exec_res["transformed_df"])
    }

@app.post("/duckdb_sql/")
async def duckdb_sql_endpoint(file: UploadFile = File(...), sql_query: str = Form(...)):
    contents = await file.read()
    df = parse_uploaded_file(file, contents)
    df_cleaned = clean_and_profile_dataframe(df)
    result_df = duckdb_engine.execute_sql_query(df_cleaned, sql_query)
    
    return {
        "status": "success",
        "sql_query": sql_query,
        "result_count": len(result_df),
        "data_preview": result_df.head(20).to_dict(orient="records")
    }

@app.post("/agent_plan/")
def agent_plan_endpoint(query: str = Form(...)):
    steps = agent_planner.decompose_query(query)
    return {
        "status": "success",
        "raw_query": query,
        "total_steps": len(steps),
        "execution_plan": steps
    }

@app.post("/clean_data/")
async def clean_data_endpoint(file: UploadFile = File(...)):
    contents = await file.read()
    df = parse_uploaded_file(file, contents)
    df_cleaned = clean_and_profile_dataframe(df)
    profile = generate_data_profile(df_cleaned)
    
    return {
        "status": "success",
        "filename": file.filename,
        "cleaned_data_preview": df_cleaned.head(10).to_dict(orient="records"),
        "data_profile": profile
    }

@app.post("/detect_anomalies/")
async def detect_anomalies_endpoint(file: UploadFile = File(...), target_column: str = Form(...)):
    contents = await file.read()
    df = parse_uploaded_file(file, contents)
    df_cleaned = clean_and_profile_dataframe(df)
    anomalies = detect_anomalies(df_cleaned, target_column)
    
    return {
        "status": "success",
        "target_column": target_column,
        "anomaly_count": len(anomalies),
        "anomalies": anomalies.to_dict(orient="records") if not anomalies.empty else []
    }

@app.post("/impute_mice/")
async def impute_mice_endpoint(file: UploadFile = File(...)):
    contents = await file.read()
    df = parse_uploaded_file(file, contents)
    df_imputed = impute_missing_values(df, method='mice')
    return {
        "status": "success",
        "method": "MICE (IterativeImputer)",
        "imputed_data_preview": df_imputed.head(10).to_dict(orient="records"),
        "data_profile": generate_data_profile(df_imputed)
    }

@app.post("/encode_features/")
async def encode_features_endpoint(file: UploadFile = File(...), method: str = Form("onehot")):
    contents = await file.read()
    df = parse_uploaded_file(file, contents)
    df_encoded = encode_categorical_features(df, method=method)
    return {
        "status": "success",
        "encoding_method": method,
        "encoded_columns": df_encoded.columns.tolist(),
        "encoded_data_preview": df_encoded.head(10).to_dict(orient="records")
    }

@app.post("/narrative_summary/")
async def narrative_summary_endpoint(file: UploadFile = File(...)):
    contents = await file.read()
    df = parse_uploaded_file(file, contents)
    df_cleaned = clean_and_profile_dataframe(df)
    narrative = generate_narrative_summary(df_cleaned)
    return {
        "status": "success",
        "narrative": narrative
    }

@app.get("/benchmark/")
def get_benchmark_results():
    results = run_benchmark_suite()
    return results
