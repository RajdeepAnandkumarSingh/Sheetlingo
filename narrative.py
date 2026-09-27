import re
import pandas as pd
import numpy as np
from cleaner import generate_data_profile

def generate_narrative_summary(df: pd.DataFrame) -> dict:
    """
    Generates automated descriptive natural language insights highlighting key statistical distributions,
    correlations, skewness, anomalies, and data quality scores upon dataset ingestion.
    """
    profile = generate_data_profile(df)
    total_rows = profile["total_rows"]
    total_cols = profile["total_columns"]
    hygiene = profile["data_hygiene_score"]
    
    num_df = df.select_dtypes(include=[np.number])
    cat_df = df.select_dtypes(exclude=[np.number])
    
    insights = []
    
    # 1. Dataset Overview Insight
    insights.append(
        f"The dataset has **{total_rows:,} rows** and **{total_cols} columns**. "
        f"Completeness is **{hygiene}%** (the share of cells that are not missing)."
    )
    
    # 2. Missing Value Analysis
    missing_cols = [col for col, pct in profile["null_percentages"].items() if pct > 0]
    if missing_cols:
        top_missing = sorted(profile["null_percentages"].items(), key=lambda x: x[1], reverse=True)[0]
        insights.append(
            f"Missing data detected in **{len(missing_cols)} column(s)**. Highest missingness is in **'{top_missing[0]}'** with **{top_missing[1]}%** null values."
        )
    else:
        insights.append("No missing cells found. Dataset completeness is 100%.")

    # Skip identifiers and near-unique text fields so an ID or person's name is
    # not presented as a useful category insight.
    if not cat_df.empty:
        eligible_cat_cols = []
        for col in cat_df.columns:
            normalized_name = re.sub(r"[^a-z0-9]+", "_", str(col).lower()).strip("_")
            name_parts = normalized_name.split("_")
            unique_count = cat_df[col].nunique(dropna=True)
            looks_like_id = (
                any(part in {"id", "uuid", "key", "identifier"} for part in name_parts)
                or unique_count > max(20, total_rows * 0.7)
            )
            if not looks_like_id and 1 < unique_count < max(2, total_rows):
                eligible_cat_cols.append(col)

        if eligible_cat_cols:
            top_cat = eligible_cat_cols[0]
            mode_values = cat_df[top_cat].mode(dropna=True)
            if not mode_values.empty:
                mode_val = mode_values.iloc[0]
                mode_freq = int((cat_df[top_cat] == mode_val).sum())
                pct_share = round((mode_freq / max(total_rows, 1)) * 100, 1)
                insights.append(
                    f"In **'{top_cat}'**, the most common value is **'{mode_val}'** "
                    f"({mode_freq:,} records, {pct_share}% of rows)."
                )

    # 4. Numerical summary and robust IQR outlier hints
    if not num_df.empty:
        target_num = num_df.columns[0]
        mean_val = round(num_df[target_num].mean(), 2)
        std_val = round(num_df[target_num].std(), 2)
        skew_val = profile["skewness"].get(target_num, 0)
        
        skew_desc = "symmetrical"
        if skew_val > 1.0:
            skew_desc = "strongly right-skewed"
        elif skew_val < -1.0:
            skew_desc = "strongly left-skewed"
            
        insights.append(
            f"Key numerical metric **'{target_num}'** averages **{mean_val:,}** (±{std_val:,}) and exhibits a {skew_desc} distribution (skewness: {skew_val})."
        )
        
        unusual_value_counts = {}
        for col in num_df.columns:
            values = pd.to_numeric(num_df[col], errors="coerce").dropna()
            if len(values) < 4:
                continue
            first_quartile, third_quartile = values.quantile([0.25, 0.75])
            interquartile_range = third_quartile - first_quartile
            if interquartile_range <= 0:
                continue
            lower_bound = first_quartile - 1.5 * interquartile_range
            upper_bound = third_quartile + 1.5 * interquartile_range
            count = int(((values < lower_bound) | (values > upper_bound)).sum())
            if count:
                unusual_value_counts[col] = count

        if unusual_value_counts:
            outlier_summary = ", ".join(
                f"**'{col}'** ({count} record{'s' if count != 1 else ''})"
                for col, count in sorted(unusual_value_counts.items(), key=lambda item: item[1], reverse=True)[:3]
            )
            insights.append(
                f"The 1.5×IQR rule found potential unusual values in {outlier_summary}. Review these before analysis."
            )

    # 5. Correlation Summary
    if len(num_df.columns) >= 2:
        corr_matrix = num_df.corr().abs()
        np.fill_diagonal(corr_matrix.values, 0)
        max_corr = corr_matrix.unstack().sort_values(ascending=False)
        if not max_corr.empty:
            pair, high_val = max_corr.index[0], round(max_corr.iloc[0], 2)
            insights.append(
                f"Highest numerical correlation detected between **'{pair[0]}'** and **'{pair[1]}'** with Pearson r = **{high_val}**."
            )

    narrative_text = " ".join(insights)
    
    return {
        "dataset_name": "Ingested Tabular Data",
        "data_hygiene_score": hygiene,
        "key_insights": insights,
        "full_narrative": narrative_text
    }
