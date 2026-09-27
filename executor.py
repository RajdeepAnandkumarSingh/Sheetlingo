import re
import hashlib
import pandas as pd
import numpy as np
from cleaner import detect_anomalies, impute_missing_values, encode_categorical_features
from narrative import generate_narrative_summary

def execute_nlp_query(df: pd.DataFrame, intent: str, entities: dict) -> dict:
    """
    Executes computational or transformation operations on DataFrame based on intent and entities.
    Returns result metrics, transformed DataFrame, chart specification, Python code trace, narrative, and message.
    """
    df = df.copy()
    columns = entities.get("columns", [])
    operator = entities.get("operator", "==")
    all_numbers = entities.get("numbers", [])
    action_numbers = entities.get("action_numbers", [])
    cond_numbers = entities.get("cond_numbers", [])
    literals = entities.get("literal_values", [])
    raw_query = entities.get("raw_query", "")
    action_clause = entities.get("action_clause", "")
    cond_clause = entities.get("cond_clause", "")
    replacement_from = entities.get("replacement_from")
    replacement_to = entities.get("replacement_to")

    result_metric = None
    chart_spec = None
    transformed_df = df
    code_snippet = ""
    status_msg = ""

    num_cols = list(df.select_dtypes(include=[np.number]).columns)
    numeric_data = pd.DataFrame(index=df.index)
    for column in df.columns:
        if column in num_cols:
            numeric_data[column] = pd.to_numeric(df[column], errors="coerce")
            continue
        present = df[column].notna() & df[column].astype(str).str.strip().ne("")
        if not present.any():
            continue
        parsed = pd.to_numeric(
            df[column].astype(str).str.replace(r"[$,]", "", regex=True).str.strip(),
            errors="coerce",
        )
        if parsed[present].notna().mean() >= 0.8:
            num_cols.append(column)
            numeric_data[column] = parsed
    cat_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()

    def first_mode(series):
        modes = series.mode(dropna=True)
        return modes.iloc[0] if not modes.empty else np.nan

    def aggregate_series(series, aggregation):
        if aggregation == "mode":
            return first_mode(series)
        return getattr(series, aggregation)()

    def display_value(value):
        if pd.isna(value):
            return "N/A"
        if isinstance(value, (np.integer,)):
            return int(value)
        if isinstance(value, (np.floating,)):
            return round(float(value), 4)
        return value

    target_col = entities.get("target_column") or (columns[0] if len(columns) > 0 else (num_cols[0] if num_cols else (df.columns[0] if len(df.columns) > 0 else "")))
    needs_clarification = False
    clarification_message = entities.get("clarification")

    # 1. UPDATE_VALUE / CONDITIONAL REPLACEMENT
    if intent == "UPDATE_VALUE":
        if clarification_message:
            needs_clarification = True
            status_msg = clarification_message
        elif replacement_from is not None and replacement_to is not None and target_col in df.columns:
            source_mask = df[target_col].astype(str).str.strip().str.casefold() == replacement_from.casefold()
            updated_rows_count = int(source_mask.sum())
            if updated_rows_count:
                transformed_df.loc[source_mask, target_col] = replacement_to
                code_snippet = (
                    f"transformed_df.loc[transformed_df['{target_col}'].astype(str).str.strip().str.casefold() "
                    f"== {replacement_from.casefold()!r}, '{target_col}'] = {replacement_to!r}"
                )
                status_msg = (
                    f"Replaced '{replacement_from}' with '{replacement_to}' in column '{target_col}'. "
                    f"({updated_rows_count} rows updated)"
                )
            else:
                needs_clarification = True
                clarification_message = f"I couldn't find '{replacement_from}' in '{target_col}', so I made no changes. Check the value or column name."
                status_msg = clarification_message
        elif target_col not in df.columns:
            needs_clarification = True
            clarification_message = "I couldn't identify the column to change. Name the column, for example ‘change Sales to Sells in Department’."
            status_msg = clarification_message
        else:
            is_null_target = any(w in action_clause or w in raw_query for w in ["null", "nan", "none", "empty", "blank"])
            if is_null_target:
                new_val = np.nan
            else:
                new_val = action_numbers[0] if action_numbers else (all_numbers[0] if all_numbers else None)

            cond_val = cond_numbers[0] if cond_numbers else None
            cond_col = entities.get("condition_column") or target_col
            if new_val is not None and cond_val is not None and cond_col in df.columns:
                try:
                    condition_series = pd.to_numeric(df[cond_col], errors="coerce")
                    if operator == ">":
                        cond_mask = condition_series > cond_val
                    elif operator == ">=":
                        cond_mask = condition_series >= cond_val
                    elif operator == "<":
                        cond_mask = condition_series < cond_val
                    elif operator == "<=":
                        cond_mask = condition_series <= cond_val
                    elif operator == "!=":
                        cond_mask = condition_series != cond_val
                    else:
                        cond_mask = condition_series == cond_val
                    updated_rows_count = int(cond_mask.sum())
                    transformed_df.loc[cond_mask, target_col] = new_val
                    val_display = "null (NaN)" if pd.isna(new_val) else new_val
                    status_msg = f"Set '{target_col}' to {val_display} where '{cond_col}' {operator} {cond_val}. ({updated_rows_count} rows updated)"
                    code_snippet = f"transformed_df.loc[pd.to_numeric(transformed_df['{cond_col}'], errors='coerce') {operator} {cond_val}, '{target_col}'] = {repr(new_val)}"
                except (TypeError, ValueError):
                    needs_clarification = True
                    clarification_message = f"The condition in '{cond_col}' isn't numeric, so I made no changes."
                    status_msg = clarification_message
            elif new_val is not None and (is_null_target or any(w in raw_query for w in ["all", "every", "entire"])):
                transformed_df[target_col] = new_val
                val_display = "null (NaN)" if pd.isna(new_val) else new_val
                status_msg = f"Set every value in '{target_col}' to {val_display}."
                code_snippet = f"transformed_df['{target_col}'] = {repr(new_val)}"
            else:
                needs_clarification = True
                clarification_message = "I couldn't safely tell which values to change, so I made no changes. Try ‘replace Sales with Sells in Department’ or specify a condition."
                status_msg = clarification_message

    # 2. SORT
    elif intent == "SORT":
        ascending = not any(w in raw_query for w in ["descending", "desc", "high to low", "highest first"])
        transformed_df = df.sort_values(by=target_col, ascending=ascending)
        sort_order = "ascending" if ascending else "descending"
        code_snippet = f"transformed_df = df.sort_values(by='{target_col}', ascending={ascending})"
        status_msg = f"Sorted dataset by column '{target_col}' in {sort_order} order."

    # 3. TOP_N / BOTTOM_N
    elif intent == "TOP_N":
        n_count = all_numbers[0] if all_numbers else 5
        is_bottom = any(w in raw_query for w in ["bottom", "lowest", "least"])
        if target_col in num_cols:
            if is_bottom:
                transformed_df = df.nsmallest(int(n_count), target_col)
            else:
                transformed_df = df.nlargest(int(n_count), target_col)
            code_snippet = f"transformed_df = df.n{'smallest' if is_bottom else 'largest'}({int(n_count)}, '{target_col}')"
        else:
            transformed_df = df.head(int(n_count))
            code_snippet = f"transformed_df = df.head({int(n_count)})"
        status_msg = f"Displayed top {int(n_count)} records based on '{target_col}'."

    # 4. GROUP_BY
    elif intent == "GROUP_BY":
        group_col = entities.get("group_by_column")
        agg_func = entities.get("aggregation")
        value_col = entities.get("aggregation_column")
        aggregate_all = entities.get("all_numeric_columns", False)
        if group_col not in df.columns and columns:
            group_col = next((column for column in columns if column in df.columns and column not in num_cols), None)
        if value_col not in num_cols and len(columns) > 1:
            value_col = next((column for column in columns if column in num_cols), None)
        if not agg_func:
            if any(word in raw_query for word in ("average", "avg", "mean")):
                agg_func = "mean"
            elif "median" in raw_query:
                agg_func = "median"
            elif "mode" in raw_query:
                agg_func = "mode"
            elif any(word in raw_query for word in ("sum", "total")):
                agg_func = "sum"
            elif any(word in raw_query for word in ("minimum", "min")):
                agg_func = "min"
            elif any(word in raw_query for word in ("maximum", "max")):
                agg_func = "max"
            elif "count" in raw_query:
                agg_func = "count"
        if group_col not in df.columns:
            needs_clarification = True
            clarification_message = "I couldn't identify the grouping column. Try ‘average Salary by Department’ and use a column name from your sheet."
            status_msg = clarification_message
        elif not agg_func:
            needs_clarification = True
            clarification_message = "Name a calculation for the groups, such as sum, average, median, mode, minimum, maximum, or count."
            status_msg = clarification_message
        elif aggregate_all:
            if not num_cols:
                needs_clarification = True
                clarification_message = "I couldn't find numeric columns to summarize. Check that the spreadsheet values are numbers."
                status_msg = clarification_message
            else:
                source = numeric_data[num_cols]
                if agg_func == "count":
                    grouped = source.groupby(df[group_col], dropna=False).count()
                else:
                    grouped = source.groupby(df[group_col], dropna=False).agg(lambda series: aggregate_series(series, agg_func))
                transformed_df = grouped.reset_index()
                code_snippet = f"transformed_df = numeric_data.groupby(df['{group_col}']).agg({agg_func!r}).reset_index()"
                status_msg = f"Grouped by '{group_col}' and calculated {agg_func.upper()} for all {len(num_cols)} numeric columns."
        elif agg_func == "count" and not value_col:
            transformed_df = df.groupby(group_col, dropna=False).size().reset_index(name="row_count")
            code_snippet = f"transformed_df = df.groupby('{group_col}', dropna=False).size().reset_index(name='row_count')"
            status_msg = f"Counted rows in each '{group_col}' group."
        elif value_col not in num_cols:
            needs_clarification = True
            clarification_message = "I couldn't identify the numeric column to calculate. Try ‘average Salary by Department’ or say ‘average all numeric columns by Department’."
            status_msg = clarification_message
        else:
            grouped_values = numeric_data.groupby(df[group_col], dropna=False)[value_col]
            if agg_func == "count":
                grouped = grouped_values.count().reset_index(name=f"{value_col}_count")
            else:
                grouped = grouped_values.agg(lambda series: aggregate_series(series, agg_func)).reset_index(name=f"{agg_func}_{value_col}")
            transformed_df = grouped
            code_snippet = f"transformed_df = numeric_data.groupby(df['{group_col}'])['{value_col}'].{agg_func}().reset_index()"
            status_msg = f"Grouped by '{group_col}' and calculated {agg_func.upper()} of '{value_col}'."

    # 5. AGGREGATIONS (SUM, AVERAGE, MEDIAN, MODE, MIN, MAX, COUNT)
    elif intent in ["SUM", "AVERAGE", "MEDIAN", "MODE", "MIN", "MAX", "COUNT"]:
        aggregation = "mean" if intent == "AVERAGE" else intent.lower()
        requested_aggregations = entities.get("aggregations") or [aggregation]
        value_col = entities.get("aggregation_column")
        if value_col not in num_cols and target_col in num_cols:
            value_col = target_col
        aggregate_all = entities.get("all_numeric_columns", False)
        if aggregate_all:
            if not num_cols:
                needs_clarification = True
                clarification_message = "I couldn't find numeric columns to summarize. Check that the spreadsheet values are numbers."
                status_msg = clarification_message
            else:
                if len(requested_aggregations) > 1:
                    summary_rows = []
                    for column in num_cols:
                        row = {"column": column}
                        row.update({
                            "mean (average)" if item == "mean" else item: display_value(
                                aggregate_series(numeric_data[column], item)
                            )
                            for item in requested_aggregations
                        })
                        summary_rows.append(row)
                    transformed_df = pd.DataFrame(summary_rows)
                    metric_name = ", ".join(item.upper() for item in requested_aggregations)
                    result_metric = {"type": "table", "metric": metric_name, "values": summary_rows}
                    code_snippet = f"result = numeric_data[{num_cols!r}].agg({requested_aggregations!r})"
                    status_msg = f"Calculated {metric_name} for each of the {len(num_cols)} numeric columns."
                else:
                    values = {
                        column: display_value(aggregate_series(numeric_data[column], aggregation))
                        for column in num_cols
                    }
                    transformed_df = pd.DataFrame([values])
                    result_metric = {"type": "table", "metric": intent, "values": values}
                    code_snippet = f"result = numeric_data[{num_cols!r}].agg({aggregation!r})"
                    status_msg = f"Calculated {intent.upper()} for all {len(num_cols)} numeric columns."
        elif intent == "COUNT" and not value_col:
            val = int(len(df))
            result_metric = {"metric": intent, "target_column": "rows", "value": val}
            code_snippet = "result = len(df)"
            status_msg = f"Counted {val} rows."
        elif value_col in num_cols:
            val = display_value(aggregate_series(numeric_data[value_col], aggregation))
            result_metric = {"metric": intent, "target_column": value_col, "value": val}
            code_snippet = f"result = numeric_data[{value_col!r}].{aggregation}()"
            status_msg = f"Calculated {intent.upper()} of '{value_col}': {val}"
        elif target_col in df.columns and intent == "MODE":
            val = display_value(first_mode(df[target_col]))
            result_metric = {"metric": intent, "target_column": target_col, "value": val}
            code_snippet = f"result = df[{target_col!r}].mode().iloc[0]"
            status_msg = f"Calculated MODE of '{target_col}': {val}"
        else:
            needs_clarification = True
            clarification_message = "I couldn't identify a numeric column. Name one, such as ‘average Salary’, or say ‘average all numeric columns’."
            status_msg = clarification_message

    # 6. SHOW_NULLS
    elif intent == "SHOW_NULLS":
        if columns:
            col_target = columns[0]
            transformed_df = df[df[col_target].isna()]
            code_snippet = f"transformed_df = df[df['{col_target}'].isna()]"
            status_msg = f"Filtered {len(transformed_df)} rows with missing values in column '{col_target}'."
        else:
            transformed_df = df[df.isna().any(axis=1)]
            code_snippet = "transformed_df = df[df.isna().any(axis=1)]"
            status_msg = f"Filtered {len(transformed_df)} rows containing missing cells across any column."

    # 7. FILTER
    elif intent == "FILTER":
        parsed_conditions = entities.get("conditions", [])
        val_to_compare = entities.get("condition_value")
        condition_col = entities.get("condition_column") or target_col
        if not parsed_conditions and val_to_compare is not None:
            parsed_conditions = [{
                "column": condition_col,
                "operator": operator,
                "value": val_to_compare,
                "exact": entities.get("condition_is_exact", False),
            }]
        if not parsed_conditions and val_to_compare is None:
            fallback_value = cond_numbers[0] if cond_numbers else (all_numbers[0] if all_numbers else (literals[0] if literals else None))
            if fallback_value is not None and condition_col in df.columns:
                parsed_conditions = [{"column": condition_col, "operator": operator, "value": fallback_value, "exact": False}]

        if entities.get("return_all_rows") and not clarification_message:
            transformed_df = df.copy()
            status_msg = f"Showing all {len(transformed_df)} rows."
            code_snippet = "transformed_df = df.copy()"
        elif clarification_message:
            needs_clarification = True
            status_msg = clarification_message
        elif parsed_conditions and all(condition["column"] in df.columns for condition in parsed_conditions):
            combined_mask = pd.Series(True, index=df.index)
            readable_conditions = []
            for condition in parsed_conditions:
                condition_col = condition["column"]
                condition_operator = condition.get("operator", "==")
                val_to_compare = condition["value"]
                series = df[condition_col]
                if condition_operator in {">", ">=", "<", "<="}:
                    numeric_series = pd.to_numeric(series, errors="coerce")
                    if condition_operator == ">":
                        condition_mask = numeric_series > val_to_compare
                    elif condition_operator == ">=":
                        condition_mask = numeric_series >= val_to_compare
                    elif condition_operator == "<":
                        condition_mask = numeric_series < val_to_compare
                    else:
                        condition_mask = numeric_series <= val_to_compare
                elif condition_operator == "!=":
                    condition_mask = series.astype(str).str.strip().str.casefold() != str(val_to_compare).strip().casefold()
                elif condition.get("exact"):
                    condition_mask = series.astype(str).str.strip().str.casefold() == str(val_to_compare).strip().casefold()
                elif pd.api.types.is_numeric_dtype(series) or isinstance(val_to_compare, (int, float)):
                    condition_mask = pd.to_numeric(series, errors="coerce") == val_to_compare
                else:
                    pattern = r"\b" + re.escape(str(val_to_compare)) + r"\b"
                    condition_mask = series.astype(str).str.contains(pattern, case=False, na=False)
                    if not condition_mask.any():
                        condition_mask = series.astype(str).str.contains(str(val_to_compare), case=False, na=False)
                combined_mask &= condition_mask.fillna(False)
                readable_conditions.append(f"{condition_col} {condition_operator} {val_to_compare}")

            transformed_df = df[combined_mask]
            code_snippet = "transformed_df = df[combined_mask]  # all listed conditions are combined with AND"
            status_msg = f"Showing {len(transformed_df)} rows matching all conditions: " + " AND ".join(readable_conditions) + "."
        else:
            needs_clarification = True
            clarification_message = "I couldn't identify what to filter by, so I left the dataset unchanged. Name a column and value, such as ‘show rows where Department is Sales’."
            status_msg = clarification_message

    # 8. PLOT
    elif intent == "PLOT":
        if len(columns) >= 2:
            x_col, y_col = columns[0], columns[1]
        elif len(columns) == 1:
            x_col = cat_cols[0] if cat_cols else df.columns[0]
            y_col = columns[0]
        else:
            x_col = cat_cols[0] if cat_cols else df.columns[0]
            y_col = num_cols[0] if num_cols else df.columns[-1]

        chart_spec = {
            "type": "bar",
            "x": x_col,
            "y": y_col,
            "title": f"NLP Automated Visualization: {y_col} by {x_col}"
        }
        code_snippet = f"fig = px.bar(df, x='{x_col}', y='{y_col}', title='{chart_spec['title']}')"
        status_msg = f"Generated Plotly chart specification for '{y_col}' grouped by '{x_col}'."

    # 9. DROP_NULLS & IMPUTE
    elif intent == "DROP_NULLS":
        initial_len = len(df)
        transformed_df = df.dropna()
        dropped_count = initial_len - len(transformed_df)
        code_snippet = "transformed_df = df.dropna()"
        status_msg = f"Successfully dropped {dropped_count} rows with missing values."

    elif intent == "IMPUTE_KNN":
        use_mice = "mice" in str(entities).lower() or "chained" in str(entities).lower()
        method = 'mice' if use_mice else 'knn'
        transformed_df = impute_missing_values(df, method=method)
        code_snippet = f"transformed_df = impute_missing_values(df, method='{method}')"
        status_msg = f"Successfully imputed missing values using {method.upper()} algorithm."

    # 10. ANOMALY
    elif intent == "ANOMALY":
        anomalies = detect_anomalies(df, target_col)
        transformed_df = anomalies
        code_snippet = f"anomalies = detect_anomalies(df, target_col='{target_col}', contamination=0.05)"
        status_msg = f"Detected {len(anomalies)} statistical outliers in column '{target_col}' using Isolation Forest."

    # 11. DROP_COL
    elif intent == "DROP_COL":
        if target_col in df.columns:
            transformed_df = df.drop(columns=[target_col])
            code_snippet = f"transformed_df = df.drop(columns=['{target_col}'])"
            status_msg = f"Successfully removed column '{target_col}'."
        else:
            status_msg = f"Column '{target_col}' not found in DataFrame."

    # 12. ADD_COL
    elif intent == "ADD_COL":
        new_col_name = entities.get("new_column_name")
        group_col = entities.get("group_by_column")
        source_col = entities.get("aggregation_column")
        aggregation = entities.get("aggregation")
        constant = entities.get("column_constant")
        formula = entities.get("column_formula")

        if not new_col_name:
            name_match = re.search(
                r"\b(?:add|create|insert|new)\s+(?:a\s+)?(?:new\s+)?column\s+(?:called\s+|named\s+|name\s+)?"
                r"(?P<name>.+?)(?=\s+(?:with|set|initialized)\b|$)",
                raw_query,
                re.IGNORECASE,
            )
            if name_match:
                new_col_name = name_match.group("name").strip(" \t'\"`").title()
        if constant is None and all_numbers and not entities.get("aggregation"):
            if any(word in raw_query for word in ("value", "equal", "set to", "initialized")):
                constant = all_numbers[0]

        if entities.get("operation_clarification"):
            needs_clarification = True
            clarification_message = entities["operation_clarification"]
            status_msg = clarification_message
        elif not new_col_name:
            needs_clarification = True
            clarification_message = "Name the new column, for example ‘add a column named Bonus with value 500’."
            status_msg = clarification_message
        elif new_col_name in df.columns:
            needs_clarification = True
            clarification_message = f"A column named '{new_col_name}' already exists. Choose a different name."
            status_msg = clarification_message
        elif aggregation and group_col in df.columns and source_col in num_cols:
            values = numeric_data.groupby(df[group_col], dropna=False)[source_col].transform(
                lambda series: aggregate_series(series, aggregation)
            )
            transformed_df[new_col_name] = values
            code_snippet = f"transformed_df[{new_col_name!r}] = numeric_data.groupby(df[{group_col!r}])[{source_col!r}].transform({aggregation!r})"
            status_msg = f"Added '{new_col_name}' with each row's {aggregation.upper()} of '{source_col}' within its '{group_col}' group."
        elif constant is not None:
            numeric_constant = pd.to_numeric(str(constant).replace(",", "").replace("$", "").strip(), errors="coerce")
            val_to_assign = numeric_constant if not pd.isna(numeric_constant) else constant
            transformed_df[new_col_name] = val_to_assign
            code_snippet = f"transformed_df[{new_col_name!r}] = {val_to_assign!r}"
            status_msg = f"Added column '{new_col_name}' with value {val_to_assign} for every row."
        elif formula:
            normalized_formula = re.sub(r"[_\-]+", " ", str(formula).strip().strip("'\"`"))
            arithmetic_match = re.fullmatch(
                r"\s*(.+?)\s*(\+|\-|\*|/|plus|minus|times|multiplied by|divided by)\s*(.+?)\s*",
                normalized_formula,
                flags=re.IGNORECASE,
            )

            def formula_operand(operand):
                cleaned_operand = operand.strip().strip("'\"`")
                normalized_operand = re.sub(r"\s+", " ", cleaned_operand.lower())
                for column in sorted(df.columns, key=lambda value: len(str(value)), reverse=True):
                    normalized_column = re.sub(r"[_\-]+", " ", str(column).lower()).strip()
                    if normalized_operand == normalized_column and column in num_cols:
                        return numeric_data[column]
                number = pd.to_numeric(cleaned_operand.replace(",", "").replace("$", ""), errors="coerce")
                return None if pd.isna(number) else number

            if arithmetic_match:
                left = formula_operand(arithmetic_match.group(1))
                right = formula_operand(arithmetic_match.group(3))
                operation = arithmetic_match.group(2).lower()
                operation = {"plus": "+", "minus": "-", "times": "*", "multiplied by": "*", "divided by": "/"}.get(operation, operation)
                if left is not None and right is not None:
                    if operation == "+": values = left + right
                    elif operation == "-": values = left - right
                    elif operation == "*": values = left * right
                    else: values = left / right
                    transformed_df[new_col_name] = values
                    code_snippet = f"transformed_df[{new_col_name!r}] = {arithmetic_match.group(1).strip()!r} {operation} {arithmetic_match.group(3).strip()!r}"
                    status_msg = f"Added calculated column '{new_col_name}'."
                else:
                    needs_clarification = True
            else:
                needs_clarification = True

            if needs_clarification:
                clarification_message = "I couldn't safely understand that column formula. Try a numeric value, ‘average Salary by Department’, or a simple formula such as ‘Salary times 0.1’."
                status_msg = clarification_message
        else:
            needs_clarification = True
            clarification_message = "Tell me how to fill the new column, for example ‘average Salary by Department’ or ‘with value 500’."
            status_msg = clarification_message

    # 13. ADD_ROW
    elif intent == "ADD_ROW":
        row_values = entities.get("row_values", {})
        if not row_values and not entities.get("operation_clarification"):
            # Legacy-style commands can still be mapped when they name each
            # field in the request (for example, "name Alex salary 60000").
            aliases = {
                "name": ("name", "employee", "person"),
                "department": ("department", "dept", "team", "division"),
                "age": ("age", "years old"),
                "salary": ("salary", "pay", "wage", "earnings"),
            }
            for column in df.columns:
                normalized_column = re.sub(r"[^a-z0-9]+", " ", str(column).lower()).strip()
                column_aliases = aliases.get(normalized_column, (normalized_column,))
                for alias in column_aliases:
                    if alias not in raw_query:
                        continue
                    next_labels = sorted({item for values in aliases.values() for item in values if item != alias}, key=len, reverse=True)
                    next_label_pattern = "|".join(re.escape(item) for item in next_labels)
                    if column in num_cols:
                        match = re.search(r"\b" + re.escape(alias) + r"\s*(?:is|:|=|of)?\s*([$]?[-\d][\d,]*(?:\.\d+)?)", raw_query)
                    else:
                        match = re.search(
                            r"\b" + re.escape(alias) + r"\s+(?:is\s+)?(?P<value>.+?)"
                            + (r"(?=\s+(?:" + next_label_pattern + r")\b|$)" if next_label_pattern else r"$"),
                            raw_query,
                        )
                    if match:
                        parsed_value = match.group(1) if column in num_cols else match.group("value")
                        row_values[column] = str(parsed_value).strip().strip("$")
                        break
        if entities.get("operation_clarification"):
            needs_clarification = True
            clarification_message = entities["operation_clarification"]
            status_msg = clarification_message
        elif not row_values:
            needs_clarification = True
            clarification_message = "Give row values with column labels, for example ‘add a row with Name: Alex Lee, Department: Sales, Age: 28, Salary: 70000’."
            status_msg = clarification_message
        else:
            new_row = {column: pd.NA for column in df.columns}
            for column, raw_value in row_values.items():
                if column not in df.columns:
                    continue
                if column in num_cols:
                    numeric_value = pd.to_numeric(
                        str(raw_value).replace(",", "").replace("$", "").strip(),
                        errors="coerce",
                    )
                    if pd.isna(numeric_value):
                        needs_clarification = True
                        clarification_message = f"'{raw_value}' is not a number for '{column}'. I didn't add the row."
                        break
                    new_row[column] = numeric_value
                else:
                    new_row[column] = raw_value
            if not needs_clarification:
                transformed_df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
                code_snippet = "transformed_df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)"
                status_msg = f"Added one row with {len(row_values)} provided values. Unspecified cells are blank. Total rows: {len(transformed_df)}."
            else:
                status_msg = clarification_message

    # 12. CORRELATION & SUMMARY
    elif intent == "CORRELATION":
        numeric_df = df.select_dtypes(include=[np.number])
        if not numeric_df.empty:
            corr_matrix = numeric_df.corr().round(3).to_dict()
            result_metric = {"type": "correlation", "matrix": corr_matrix}
            code_snippet = "corr_matrix = df.select_dtypes(include=[np.number]).corr()"
            status_msg = "Computed Pearson correlation matrix for numeric columns."
        else:
            status_msg = "No numerical columns available for correlation analysis."

    elif intent == "SUMMARY":
        code_snippet = "summary = df.describe(include='all')"
        status_msg = "Generated summary dataset statistical profile."

    narrative = generate_narrative_summary(transformed_df)

    return {
        "status": "success",
        "intent": intent,
        "status_message": status_msg,
        "interpretation": status_msg,
        "needs_clarification": needs_clarification,
        "clarification_message": clarification_message,
        "result_metric": result_metric,
        "chart_spec": chart_spec,
        "code_snippet": code_snippet,
        "transformed_df": transformed_df,
        "narrative_summary": narrative
    }


with open(__file__, "rb") as _source_file:
    SHEETLINGO_SOURCE_DIGEST = hashlib.sha256(_source_file.read()).hexdigest()
