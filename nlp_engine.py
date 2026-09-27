import re
import hashlib
import spacy
import numpy as np
import pandas as pd
from spacy.matcher import PhraseMatcher
from rapidfuzz import process, fuzz
from word2number import w2n

# Load spaCy NLP dependency parser
try:
    nlp = spacy.load("en_core_web_sm")
except Exception:
    import spacy.cli
    try:
        spacy.cli.download("en_core_web_sm")
        nlp = spacy.load("en_core_web_sm")
    except Exception:
        # Column-aware rules still work with spaCy tokenization if the model
        # cannot be downloaded in an offline or restricted environment.
        nlp = spacy.blank("en")

# Load Sentence Transformers for Deep Semantic Vector Matching. Keep an existing
# model when this module is reloaded after a code update in the running app.
if globals().get("semantic_model") is not None:
    semantic_model = globals()["semantic_model"]
    try:
        from sentence_transformers import util
    except Exception:
        util = None
else:
    try:
        from sentence_transformers import SentenceTransformer, util
        semantic_model = SentenceTransformer('all-MiniLM-L6-v2')
    except Exception:
        semantic_model = None

# Comprehensive Operational Intent Concept Matrix
INTENT_CONCEPT_DESCRIPTIONS = {
    "SUM": [
        "calculate total sum of numeric column",
        "find total overall sum revenue expenditure quantity",
        "add up total total amount"
    ],
    "AVERAGE": [
        "calculate average mean value of numeric column",
        "find mean score price salary age earnings pay",
        "what is the average mean"
    ],
    "MEDIAN": [
        "calculate find median middle value of numeric column",
        "what is the median salary age price earnings score"
    ],
    "MODE": [
        "find mode most frequent common popular value of column",
        "what is the mode of department status city category"
    ],
    "MIN": [
        "find minimum lowest smallest value",
        "what is the lowest cost price score"
    ],
    "MAX": [
        "find maximum highest peak largest value",
        "what is the highest salary price revenue score"
    ],
    "COUNT": [
        "count total number of rows records items",
        "how many records rows employees entries are there"
    ],
    "FILTER": [
        "filter search find list retrieve display show rows where condition equals or matches value",
        "need the data of person name employee region department status",
        "show rows with salary age price above below equal to"
    ],
    "UPDATE_VALUE": [
        "fill set replace update change modify assign make convert value to new value or null",
        "in age column where age is above 40 replace it with 20",
        "if salary is greater than 80000 make it null",
        "replace bad value with new value or empty null nan",
        "change status department to active engineering",
        "make it null empty nan none where condition"
    ],
    "SORT": [
        "sort order rank dataset by column in ascending or descending high to low order",
        "order records by salary date age name"
    ],
    "TOP_N": [
        "show top N highest records",
        "show bottom N lowest records",
        "top 5 highest salaries or bottom 3 scores"
    ],
    "GROUP_BY": [
        "group by category and compute aggregate breakdown average sum count per department region team each",
        "give me a breakdown of average earnings for each team department",
        "average salary by department group"
    ],
    "SHOW_NULLS": [
        "show display find list missing cells empty rows null values na blank entries",
        "check missing data in column"
    ],
    "DROP_NULLS": [
        "drop remove delete purge empty missing null na rows records"
    ],
    "IMPUTE_KNN": [
        "fill impute estimate missing empty null values using KNN or MICE iterative imputer algorithm"
    ],
    "ANOMALY": [
        "detect find identify statistical outliers anomalies isolation forest in price salary age"
    ],
    "DROP_COL": [
        "remove drop delete column attribute field"
    ],
    "ADD_COL": [
        "add create insert new column attribute field with default or calculated value",
        "add column tax with value 500",
        "create new column bonus"
    ],
    "ADD_ROW": [
        "add insert append new row record record entry",
        "add row with name john salary 50000"
    ],
    "RENAME_COL": [
        "rename change column header name to new name"
    ],
    "PLOT": [
        "plot draw visualize create bar chart scatter plot graph distribution line chart"
    ],
    "CORRELATION": [
        "show compute calculate pearson correlation matrix heatmap between numeric columns"
    ],
    "SUMMARY": [
        "describe summarize display statistical profile overview of dataset"
    ]
}

# Pre-compute semantic vector embeddings for concept descriptions if model loaded
INTENT_EMBEDDINGS = {}
if semantic_model is not None:
    for intent_label, phrases in INTENT_CONCEPT_DESCRIPTIONS.items():
        embeddings = semantic_model.encode(phrases, convert_to_tensor=True)
        INTENT_EMBEDDINGS[intent_label] = embeddings


def infer_numeric_columns(dataframe):
    """Include numeric columns stored as text, as often happens in spreadsheets."""
    numeric_columns = list(dataframe.select_dtypes(include=[np.number]).columns)
    for column in dataframe.columns:
        if column in numeric_columns:
            continue
        values = dataframe[column]
        present = values.notna() & values.astype(str).str.strip().ne("")
        if not present.any():
            continue
        cleaned = values.astype(str).str.replace(r"[$,]", "", regex=True).str.strip()
        parsed = pd.to_numeric(cleaned, errors="coerce")
        if parsed[present].notna().mean() >= 0.8:
            numeric_columns.append(column)
    return numeric_columns

class AdvancedNLPEngine:
    def __init__(self):
        self.semantic_model = semantic_model
        self.intent_embeddings = INTENT_EMBEDDINGS

    def predict_intent(self, query: str) -> str:
        """
        Predicts intent using Deep Semantic Vector Embeddings + Syntactic Rule Guards.
        """
        if not query or not query.strip():
            return "SUMMARY"
            
        q_clean = query.lower().strip()

        # Add Column / Add Row Priority Guards
        if re.search(r"\b(?:add|create|insert|append)\s+(?:(?:a|an|new)\s+)?(?:new\s+)?column\b", q_clean) or "new column" in q_clean:
            return "ADD_COL"
        if re.search(r"\b(?:add|insert|append)\s+(?:(?:a|an|new)\s+)?(?:new\s+)?row\b", q_clean) or "new row" in q_clean:
            return "ADD_ROW"

        aggregate_match = re.search(r"\b(sum|total|average|avg|mean|median|mode|minimum|min|maximum|max|count)\b", q_clean)
        grouping_match = re.search(r"\b(?:group(?:ed)?\s+by|by|per|for each|for every|in each|based on)\b", q_clean)
        if aggregate_match and grouping_match:
            return "GROUP_BY"
        if aggregate_match:
            aggregate = aggregate_match.group(1)
            if aggregate == "median":
                return "MEDIAN"
            if aggregate == "mode":
                return "MODE"
            if aggregate in {"average", "avg", "mean"}:
                return "AVERAGE"
            if aggregate in {"minimum", "min"}:
                return "MIN"
            if aggregate in {"maximum", "max"}:
                return "MAX"
            if aggregate == "count":
                return "COUNT"
            return "SUM"

        # Anomaly Detection Priority Guard
        if any(w in q_clean for w in ["anomaly", "anomalies", "outlier", "outliers"]):
            return "ANOMALY"

        # Explicit Mutation Guard (including 'make it null', 'replace', 'set', 'update', 'change', 'convert')
        if any(w in q_clean for w in ["replace", "fill ", "set ", "update ", "change ", "make ", "convert ", "turn "]) and not any(w in q_clean for w in ["fill missing", "impute"]):
            return "UPDATE_VALUE"

        # Direct High-Priority Search/Retrieval Guard
        if any(q_clean.startswith(w) for w in ["need ", "show ", "find ", "get ", "search ", "list ", "display ", "filter ", "exclude "]):
            if not any(w in q_clean for w in ["replace", "fill ", "set ", "update ", "change ", "make ", "sort", "plot", "chart", "drop", "impute", "top ", "highest ", "bottom ", "lowest "]):
                if any(w in q_clean for w in ["missing", "null", "empty", "blank"]):
                    return "SHOW_NULLS"
                return "FILTER"

        # Group By / Breakdown heuristic
        if "breakdown" in q_clean or "grouped by" in q_clean or "per " in q_clean or "for each" in q_clean:
            if "plot" not in q_clean and "chart" not in q_clean and "graph" not in q_clean:
                return "GROUP_BY"

        # If Sentence Transformers is active, compute dense semantic similarity
        if self.semantic_model is not None and self.intent_embeddings:
            try:
                q_emb = self.semantic_model.encode(query, convert_to_tensor=True)
                best_intent = "FILTER"
                highest_score = -1.0

                for intent_label, phrase_embs in self.intent_embeddings.items():
                    sim_scores = util.cos_sim(q_emb, phrase_embs)
                    max_sim = float(sim_scores.max())
                    if max_sim > highest_score:
                        highest_score = max_sim
                        best_intent = intent_label

                if highest_score > 0.35:
                    return best_intent
            except Exception:
                pass

        # Robust Rule-Based Fallback
        if "sort" in q_clean or "order by" in q_clean:
            return "SORT"
        if ("top " in q_clean or "highest " in q_clean or "bottom " in q_clean or "lowest " in q_clean) and any(c.isdigit() for c in q_clean):
            return "TOP_N"
        if "missing" in q_clean or "null" in q_clean or "empty" in q_clean:
            if "drop" in q_clean or "delete" in q_clean:
                return "DROP_NULLS"
            if "impute" in q_clean or "fill" in q_clean or "mice" in q_clean or "knn" in q_clean:
                return "IMPUTE_KNN"
            return "SHOW_NULLS"
        if "plot" in q_clean or "chart" in q_clean or "graph" in q_clean or "draw" in q_clean:
            return "PLOT"
        if "corr" in q_clean:
            return "CORRELATION"

        return "FILTER"

    def parse_numbers_and_words(self, text: str) -> list:
        """
        Parses digit numbers, verbal word numbers ('twenty', 'forty', '50k', '70 thousand'), and decimal floats.
        """
        found_numbers = []
        if not text:
            return found_numbers

        digits = re.findall(r'-?\d+\.?\d*', text)
        for d in digits:
            try:
                num = float(d) if '.' in d else int(d)
                found_numbers.append(num)
            except ValueError:
                pass

        words = re.findall(r'\b[a-zA-Z]+\b', text.lower())
        for i in range(len(words)):
            chunk = words[i]
            if chunk not in ["a", "an", "the", "in", "of", "to", "for", "with", "where", "above", "below"]:
                try:
                    val = w2n.word_to_num(chunk)
                    if val not in found_numbers and val > 0:
                        found_numbers.append(val)
                except ValueError:
                    pass

            if i < len(words) - 1:
                bigram = f"{words[i]} {words[i+1]}"
                try:
                    val_bi = w2n.word_to_num(bigram)
                    if val_bi not in found_numbers and val_bi > 0:
                        found_numbers.append(val_bi)
                except ValueError:
                    pass

        return found_numbers

    def extract_semantic_clauses(self, query_clean: str):
        """
        Splits natural language query into Action Clause and Condition Clause regardless of sentence ordering.
        Handles both 'IF condition MAKE action' and 'MAKE action WHERE condition'.
        """
        cond_clause = ""
        action_clause = query_clean

        # Handle 'if ... make ...' syntax
        if query_clean.startswith("if "):
            make_words = [" make ", " set ", " change ", " replace ", " turn ", " convert "]
            for m_word in make_words:
                if m_word in query_clean:
                    parts = query_clean[3:].split(m_word, 1)
                    cond_clause = parts[0]
                    action_clause = m_word.strip() + " " + parts[1]
                    return action_clause.strip(), cond_clause.strip()

        split_words = [" where ", " if ", " when ", " having ", " for rows where ", " for rows with ", " of ", " for "]
        for word in split_words:
            if word in query_clean:
                parts = query_clean.split(word, 1)
                first_part, second_part = parts[0], parts[1]
                
                action_words = [" replace ", " set ", " fill ", " change ", " update ", " assign ", " make ", " turn ", " convert "]
                found_action = False
                for a_word in action_words:
                    if a_word in second_part:
                        cond_parts = second_part.split(a_word, 1)
                        cond_clause = cond_parts[0]
                        action_clause = first_part + " " + a_word + cond_parts[1]
                        found_action = True
                        break
                
                if not found_action:
                    action_clause = first_part
                    cond_clause = second_part
                break

        return action_clause.strip(), cond_clause.strip()

    def resolve_semantic_column(self, query_tokens: list, df_columns: list) -> list:
        """
        Matches query tokens against actual DataFrame columns using exact, fuzzy, and semantic embedding similarity.
        (e.g., query 'earnings' or 'pay' -> matches column 'Salary'; query 'staff' -> matches 'Name').
        """
        found_cols = []
        if not df_columns:
            return found_cols

        col_map = {str(c).lower(): c for c in df_columns}
        
        # 1. Exact & Substring Matching
        for token in query_tokens:
            token_l = str(token).lower()
            if token_l in col_map and col_map[token_l] not in found_cols:
                found_cols.append(col_map[token_l])
            for col in df_columns:
                if str(col).lower() in token_l or token_l in str(col).lower():
                    if col not in found_cols and len(token_l) >= 3:
                        found_cols.append(col)

        # 2. Fuzzy Token Matching
        for token in query_tokens:
            token_l = str(token).lower()
            if len(token_l) >= 3 and token_l not in ["data", "show", "find", "need", "make", "null"]:
                matches = process.extract(token_l, list(col_map.keys()), scorer=fuzz.ratio, score_cutoff=70)
                for matched_col_lower, score, _ in matches:
                    orig_col = col_map[matched_col_lower]
                    if orig_col not in found_cols:
                        found_cols.append(orig_col)

        # 3. Semantic Embedding Column Synonyms Fallback
        synonym_map = {
            "pay": "Salary", "earnings": "Salary", "income": "Salary", "wage": "Salary", "compensation": "Salary",
            "staff": "Name", "worker": "Name", "employee": "Name", "person": "Name", "individual": "Name",
            "team": "Department", "division": "Department", "unit": "Department", "dept": "Department",
            "years": "Age", "old": "Age", "score": "Performance_Score", "rating": "Performance_Score"
        }
        for token in query_tokens:
            token_l = str(token).lower()
            if token_l in synonym_map:
                target_syn = synonym_map[token_l]
                for col in df_columns:
                    if str(col).lower() == target_syn.lower() or target_syn.lower() in str(col).lower():
                        if col not in found_cols:
                            found_cols.append(col)

        return found_cols

    def extract_entities(self, query: str, df_columns: list, dataframe=None) -> dict:
        original_query = query.strip()
        query_clean = query.lower().strip()
        action_clause, cond_clause = self.extract_semantic_clauses(query_clean)

        # Support categorical remapping commands such as "in department make sales = sells".
        # The left side is the existing cell value and the right side is its replacement.
        value_mapping = re.search(
            r"\b(?:make|set|change|replace|update)\s+([^=]+?)\s*=\s*(.+?)\s*$",
            query_clean,
            flags=re.IGNORECASE,
        )
        if value_mapping is None:
            value_mapping = re.search(
                r"\b(?:replace)\s+(.+?)\s+with\s+(.+?)(?:\s+in\s+.+)?\s*$|"
                r"\b(?:change|set|make|update|convert)\s+(.+?)\s+(?:to|into)\s+(.+?)(?:\s+in\s+.+)?\s*$",
                query_clean,
                flags=re.IGNORECASE,
            )
        if value_mapping and value_mapping.lastindex == 4:
            source_value = value_mapping.group(1) or value_mapping.group(3)
            target_value = value_mapping.group(2) or value_mapping.group(4)
        elif value_mapping:
            source_value, target_value = value_mapping.group(1), value_mapping.group(2)
        else:
            source_value = target_value = None
        replacement_from = source_value.strip(" \t'\"`") if source_value else None
        replacement_to = target_value.strip(" \t'\"`.,;!") if target_value else None
        if replacement_from and replacement_from.lower() in {"it", "them", "that", "those", "these"}:
            replacement_from = replacement_to = None
        
        doc_full = nlp(query_clean)
        tokens = [token.text for token in doc_full]
        lemma_terms = [token.lemma_.lower() for token in doc_full if token.lemma_ and token.lemma_ != "-PRON-"]
        lemmatized_query = " ".join(lemma_terms)

        found_cols = self.resolve_semantic_column(tokens, df_columns)
        normalized_query = re.sub(r"[_\-]+", " ", query_clean)

        # Prefer a column explicitly named by the user over incidental fuzzy matches.
        target_column = next((
            col for col in sorted(df_columns, key=lambda value: len(str(value)), reverse=True)
            if re.search(
                r"(?<!\w)" + re.escape(re.sub(r"[_\-]+", " ", str(col).lower())) + r"(?!\w)",
                normalized_query,
            )
        ), None)
        synonym_families = {
            "salary": {"salary", "salaries", "pay", "wage", "wages", "earning", "earnings", "income", "compensation"},
            "department": {"department", "departments", "team", "teams", "division", "unit", "dept", "group"},
            "age": {"age", "ages", "older", "aged", "years old", "year old"},
            "person": {"name", "names", "staff", "worker", "workers", "person", "people", "individual", "customer", "client"},
            "location": {"city", "region", "location", "place", "country", "state"},
            "date": {"date", "day", "month", "year", "time", "created", "occurred"},
            "price": {"price", "cost", "amount", "value", "revenue", "sales", "profit"},
        }

        def resolve_column_in_text(text: str, numeric_only: bool = False):
            normalized_text = re.sub(r"[_\-]+", " ", str(text).lower())
            candidates = list(df_columns)
            if numeric_only and dataframe is not None:
                numeric_names = set(infer_numeric_columns(dataframe))
                candidates = [col for col in candidates if col in numeric_names]

            explicit = []
            for col in candidates:
                phrase = re.sub(r"[_\-]+", " ", str(col).lower()).strip()
                match = re.search(r"(?<!\w)" + re.escape(phrase) + r"(?!\w)", normalized_text)
                if match:
                    explicit.append((len(phrase), match.start(), col))
            if explicit:
                return max(explicit, key=lambda item: (item[0], item[1]))[2]

            for family, aliases in synonym_families.items():
                if not any(re.search(r"(?<!\w)" + re.escape(alias) + r"(?!\w)", normalized_text) for alias in aliases):
                    continue
                family_candidates = [
                    col for col in candidates
                    if any(re.search(r"(?<!\w)" + re.escape(hint) + r"(?!\w)", re.sub(r"[_\-]+", " ", str(col).lower())) for hint in aliases | {family})
                ]
                if len(family_candidates) == 1:
                    return family_candidates[0]

            # Match small header variations such as "sleep duration hour"
            # against "Sleep_Duration_Hours". Only accept a strong, clearly
            # better candidate so a vague description does not pick a column
            # arbitrarily.
            query_phrase = " ".join(re.findall(r"[a-z0-9]+", normalized_text))
            fuzzy_matches = []
            if query_phrase:
                for column in candidates:
                    column_phrase = " ".join(
                        re.findall(r"[a-z0-9]+", re.sub(r"[_\-]+", " ", str(column).lower()))
                    )
                    score = fuzz.token_sort_ratio(query_phrase, column_phrase)
                    fuzzy_matches.append((score, column))
            fuzzy_matches.sort(key=lambda item: item[0], reverse=True)
            if fuzzy_matches and fuzzy_matches[0][0] >= 88:
                runner_up_score = fuzzy_matches[1][0] if len(fuzzy_matches) > 1 else 0
                if fuzzy_matches[0][0] - runner_up_score >= 6:
                    return fuzzy_matches[0][1]
            return None

        numeric_columns = infer_numeric_columns(dataframe) if dataframe is not None else []
        aggregate_matches = re.findall(
            r"\b(sum|total|average|avg|mean|median|mode|minimum|min|maximum|max|count)\b",
            normalized_query,
        )
        aggregation_aliases = {
            "avg": "mean", "average": "mean", "mean": "mean",
            "total": "sum", "minimum": "min", "maximum": "max",
        }
        aggregations = list(dict.fromkeys(aggregation_aliases.get(item, item) for item in aggregate_matches))
        aggregation = None
        if aggregations:
            aggregation = aggregations[0]

        all_numeric_columns = bool(re.search(
            r"\b(?:all\s+|each\s+|every\s+)?(?:numeric|numerical|number)\s+(?:columns?|fields?)\b",
            normalized_query,
        ))
        aggregate_subject_match = re.search(
            r"\b(?:sum|total|average|avg|mean|median|mode|minimum|min|maximum|max|count)\b"
            r"(?:\s+of)?\s+(?P<subject>.+?)"
            r"(?=\s+(?:by|per|for each|for every|in each|group(?:ed)? by|based on|that should be)\b|[,;.]|$)",
            normalized_query,
        )
        aggregation_column = None
        if aggregate_subject_match and not all_numeric_columns:
            aggregation_column = resolve_column_in_text(aggregate_subject_match.group("subject"), numeric_only=True)
        if aggregation_column is None and aggregation and not all_numeric_columns and target_column in numeric_columns:
            aggregation_column = target_column

        grouping_match = re.search(
            r"\b(?:based on|group(?:ed)?\s+by|for each|for every|in each|per|by)\s+"
            r"(?P<group>.+?)(?=\s+(?:with|using|and|where|that should be)\b|[,;.]|$)",
            normalized_query,
        )
        group_by_column = resolve_column_in_text(grouping_match.group("group")) if grouping_match else None

        new_column_name = None
        column_name_match = re.search(
            r"\bcolumn\s+(?:(?:called|named|name)\s+)?(?P<name>.+?)"
            r"(?=\s+(?:that\s+should\s+be|based\s+on|with\s+(?:a\s+)?value|set\s+to|equal\s+to|as|for\s+each|for\s+every|in\s+each|per|by)\b|\s*=|$)",
            original_query,
            re.IGNORECASE,
        )
        if column_name_match:
            new_column_name = re.sub(
                r"^(?:of|for)\s+", "", column_name_match.group("name").strip(" \t'\"`"), flags=re.IGNORECASE
            )

        formula_match = re.search(
            r"\b(?:as|equal to|equals?)\s+(?P<formula>.+?)\s*$|=\s*(?P<formula_equal>.+?)\s*$",
            original_query,
            re.IGNORECASE,
        )
        column_formula = (formula_match.group("formula") or formula_match.group("formula_equal")) if formula_match else None

        constant_match = re.search(
            r"\b(?:with\s+(?:a\s+)?value(?:\s+of)?|set\s+to|initialized\s+to)\s+(?P<value>.+?)\s*$",
            original_query,
            re.IGNORECASE,
        )
        column_constant = constant_match.group("value").strip(" \t'\"`") if constant_match else None

        row_values = {}
        operation_clarification = None
        row_label_columns = {}
        for column in df_columns:
            label = re.sub(r"[_\-]+", " ", str(column).lower()).strip()
            row_label_columns[label] = column
        for family, aliases in synonym_families.items():
            family_columns = [
                col for col in df_columns
                if any(
                    re.search(r"(?<!\w)" + re.escape(hint) + r"(?!\w)", re.sub(r"[_\-]+", " ", str(col).lower()))
                    for hint in aliases | {family}
                )
            ]
            if len(family_columns) == 1:
                for alias in aliases:
                    row_label_columns[alias] = family_columns[0]
        row_label_pattern = "|".join(
            re.escape(label) for label in sorted(row_label_columns, key=len, reverse=True)
        )
        row_clause = re.search(
            r"\b(?:add|insert|append)\s+(?:a\s+|an\s+)?(?:new\s+)?row\b"
            r"\s*(?:with|containing|where)?\s*(?P<values>.+)$",
            original_query,
            re.IGNORECASE,
        )
        if row_clause:
            fields = re.split(
                r",|;|\s+and\s+(?=(?:" + row_label_pattern + r")\s*(?::|=|\bis\b|\s))",
                row_clause.group("values"),
                flags=re.IGNORECASE,
            )
            for field in fields:
                field_match = re.match(r"\s*(?P<label>.+?)\s*(?::|=|\bis\b)\s*(?P<value>.+?)\s*$", field)
                if not field_match:
                    for label in sorted(row_label_columns, key=len, reverse=True):
                        natural_label_match = re.match(
                            r"\s*" + re.escape(label) + r"\s+(?:is\s+)?(?P<value>.+?)\s*$",
                            field,
                            re.IGNORECASE,
                        )
                        if natural_label_match:
                            field_match = natural_label_match
                            field_match_label = label
                            break
                    else:
                        field_match_label = None
                else:
                    field_match_label = None
                if not field_match:
                    operation_clarification = "Add a row with labeled values, for example: ‘add a row with Name: Alex Lee, Department: Sales, Age: 28, Salary: 70000’."
                    break
                label = field_label = (
                    field_match_label if field_match_label is not None else field_match.group("label").strip()
                )
                column = resolve_column_in_text(label)
                if column is None:
                    column = row_label_columns.get(re.sub(r"[_\-]+", " ", label.lower()).strip())
                if column is None:
                    normalized_label = re.sub(r"[^a-z0-9]+", " ", label.lower()).strip()
                    synonym_aliases = {
                        "name": {"name", "employee", "employee name", "person"},
                        "department": {"department", "team", "division", "dept"},
                        "age": {"age", "years old"},
                        "salary": {"salary", "pay", "wage", "earnings"},
                    }
                    for family, aliases in synonym_aliases.items():
                        if normalized_label in aliases:
                            matches = [col for col in df_columns if family in re.sub(r"[^a-z0-9]+", " ", str(col).lower()).split()]
                            if len(matches) == 1:
                                column = matches[0]
                                break
                if column is None:
                    operation_clarification = f"I couldn't match ‘{label}’ to a column. Use one of these column names: {', '.join(map(str, df_columns))}."
                    break
                row_values[column] = field_match.group("value").strip(" \t'\"`")

        if target_column is None:
            for family, aliases in synonym_families.items():
                if any(
                    re.search(r"(?<!\w)" + re.escape(alias) + r"(?!\w)", normalized_query)
                    or re.search(r"(?<!\w)" + re.escape(alias) + r"(?!\w)", lemmatized_query)
                    for alias in aliases
                ):
                    column_hints = aliases | {family}
                    candidates = [
                        col for col in df_columns
                        if any(re.search(r"(?<!\w)" + re.escape(hint) + r"(?!\w)", str(col).lower()) for hint in column_hints)
                    ]
                    if len(candidates) == 1:
                        target_column = candidates[0]
                        break
        if target_column is None and found_cols:
            target_column = found_cols[0]
        if aggregation_column is not None:
            target_column = aggregation_column

        conditions = []
        clarification = None
        return_all_rows = any(word in normalized_query for word in ["all rows", "all records", "all employees", "everyone", "everything"])

        # Match dataset values so people can filter by what they know (for example,
        # "show John from Sales") without knowing every column heading.
        if dataframe is not None and not dataframe.empty:
            all_values = []
            for col in dataframe.select_dtypes(exclude=[np.number]).columns:
                all_values.extend((col, value.strip()) for value in dataframe[col].dropna().astype(str).unique())
            complete_value_hits = [
                (col, value) for col, value in all_values if value and re.search(
                    r"(?<!\w)" + re.escape(value.lower()) + r"(?!\w)", normalized_query
                )
            ]
            categorical_hits = complete_value_hits
            if not categorical_hits:
                for col, value in all_values:
                    value_words = [word for word in re.findall(r"[\w']+", value.lower()) if len(word) >= 3]
                    if any(re.search(r"(?<!\w)" + re.escape(word) + r"(?!\w)", normalized_query) for word in value_words):
                        categorical_hits.append((col, value))
            categorical_hits = list(dict.fromkeys(categorical_hits))
            hit_values_by_column = {}
            for col, value in categorical_hits:
                hit_values_by_column.setdefault(col, set()).add(value.casefold())
            conflicting_values = any(len(values) > 1 for values in hit_values_by_column.values())
            if conflicting_values:
                clarification = "I found multiple values for the same column. Please specify one value, or say which values to include."
            elif categorical_hits:
                for col, value in categorical_hits:
                    value_pattern = r"(?<!\w)" + re.escape(value.lower()) + r"(?!\w)"
                    negative_pattern = r"\b(?:not|except|excluding|exclude|other than)\b.*" + value_pattern
                    value_operator = "!=" if re.search(negative_pattern, normalized_query) else "=="
                    conditions.append({"column": col, "operator": value_operator, "value": value, "exact": True})

        # Parse one or more numeric comparisons such as "age over forty and pay under 80k".
        numeric_condition_pattern = re.compile(
            r"(?P<operator>>=|<=|!=|==|=|>|<|greater than or equal(?: to)?|older than or equal(?: to)?|"
            r"younger than or equal(?: to)?|at least|minimum of|less than or equal(?: to)?|at most|maximum of|"
            r"greater than|older than|younger than|above|more than|exceeding|over|less than|below|under|fewer than)"
            r"\s*\$?\s*(?P<value>-?\d[\d,]*(?:\.\d+)?|[a-z]+(?:[ -]+[a-z]+){0,2})\s*"
            r"(?P<scale>thousand|million|[kKmM])?",
        )
        numeric_conditions = list(numeric_condition_pattern.finditer(normalized_query))
        if len(numeric_conditions) + len(conditions) > 1 and re.search(r"\bor\b", normalized_query):
            clarification = "I can combine conditions with ‘and’, but ‘or’ can mean different things. Please simplify the request."

        for numeric_condition in numeric_conditions:
            operator_text = numeric_condition.group("operator").lower()
            if operator_text in {">=", "greater than or equal", "greater than or equal to", "older than or equal", "older than or equal to", "at least", "minimum of"}:
                condition_operator = ">="
            elif operator_text in {"<=", "less than or equal", "less than or equal to", "younger than or equal", "younger than or equal to", "at most", "maximum of"}:
                condition_operator = "<="
            elif operator_text in {"!=", "not equal"}:
                condition_operator = "!="
            elif operator_text in {"<", "less than", "younger than", "below", "under", "fewer than"}:
                condition_operator = "<"
            elif operator_text in {"=", "=="}:
                condition_operator = "=="
            elif operator_text == "older than":
                condition_operator = ">"
            else:
                condition_operator = ">"

            raw_numeric_value = numeric_condition.group("value")
            try:
                numeric_value = float(raw_numeric_value.replace(",", ""))
            except ValueError:
                try:
                    numeric_value = float(w2n.word_to_num(raw_numeric_value))
                except ValueError:
                    parsed_word_numbers = self.parse_numbers_and_words(raw_numeric_value)
                    numeric_value = float(parsed_word_numbers[0]) if parsed_word_numbers else None
            scale = numeric_condition.group("scale")
            if scale and numeric_value is not None:
                numeric_value *= 1_000 if scale.lower() in {"k", "thousand"} else 1_000_000

            prefix = normalized_query[:numeric_condition.start("operator")]
            explicit_numeric_column = None
            explicit_mentions = []
            for col in df_columns:
                column_phrase = re.sub(r"[_\-]+", " ", str(col).lower())
                matches = list(re.finditer(r"(?<!\w)" + re.escape(column_phrase) + r"(?!\w)", prefix))
                if matches:
                    explicit_mentions.append((matches[-1].start(), col))
            if explicit_mentions:
                explicit_numeric_column = max(explicit_mentions, key=lambda item: item[0])[1]
            else:
                alias_mentions = []
                numeric_cols = set(dataframe.select_dtypes(include=[np.number]).columns) if dataframe is not None else set(df_columns)
                if operator_text in {"older than", "older than or equal", "older than or equal to", "younger than", "younger than or equal", "younger than or equal to"}:
                    age_candidates = [
                        col for col in df_columns if col in numeric_cols and
                        any(re.search(r"(?<!\w)" + re.escape(hint) + r"(?!\w)", str(col).lower()) for hint in synonym_families["age"] | {"age"})
                    ]
                    if len(age_candidates) == 1:
                        explicit_numeric_column = age_candidates[0]
                for family, aliases in synonym_families.items():
                    candidates = [
                        col for col in df_columns if col in numeric_cols and
                        any(re.search(r"(?<!\w)" + re.escape(hint) + r"(?!\w)", str(col).lower()) for hint in aliases | {family})
                    ]
                    if len(candidates) == 1:
                        for alias in aliases:
                            matches = list(re.finditer(r"(?<!\w)" + re.escape(alias) + r"(?!\w)", prefix))
                            if matches:
                                alias_mentions.append((matches[-1].start(), candidates[0]))
                if alias_mentions:
                    explicit_numeric_column = max(alias_mentions, key=lambda item: item[0])[1]

            if numeric_value is None:
                clarification = "I couldn't understand one of those numbers. Use a number such as 40 or ‘forty’."
            elif explicit_numeric_column is None:
                clarification = "I found a number but not which column it belongs to. Try ‘show rows where Age is over 40’."
            else:
                conditions.append({
                    "column": explicit_numeric_column,
                    "operator": condition_operator,
                    "value": numeric_value,
                    "exact": False,
                })

        if len(conditions) > 1 and re.search(r"\bor\b", normalized_query):
            clarification = "I can combine conditions with ‘and’, but ‘or’ can mean different things. Please simplify the request."
        if conditions:
            condition_column = conditions[0]["column"]
            condition_value = conditions[0]["value"]
            condition_is_exact = conditions[0]["exact"]
        else:
            condition_column = target_column
            condition_value = None
            condition_is_exact = False

        target_text_for_op = cond_clause if cond_clause else query_clean
        op = "=="
        if any(w in target_text_for_op for w in ["greater than or equal", "at least", "minimum of", ">="]):
            op = ">="
        elif any(w in target_text_for_op for w in ["less than or equal", "at most", "maximum of", "<="]):
            op = "<="
        elif any(w in target_text_for_op for w in ["greater than", "above", "more than", "exceeding", "over", ">"]):
            op = ">"
        elif any(w in target_text_for_op for w in ["less than", "below", "under", "fewer than", "<"]):
            op = "<"
        elif any(w in target_text_for_op for w in ["not equal", "different from", "!="]):
            op = "!="
        if conditions:
            op = conditions[0]["operator"]

        action_numbers = self.parse_numbers_and_words(action_clause)
        cond_numbers = self.parse_numbers_and_words(cond_clause) if cond_clause else []
        all_numbers = self.parse_numbers_and_words(query_clean)

        ignore_words = {
            "what", "is", "the", "where", "for", "with", "show", "find", "get", "plot", "filter",
            "average", "total", "sum", "mean", "rows", "row", "column", "columns", "data", "status",
            "equals", "greater", "than", "less", "above", "below", "under", "and", "or", "in", "by",
            "missing", "cells", "cell", "values", "value", "null", "nulls", "empty", "blank", "na",
            "nan", "none", "list", "display", "check", "detect", "anomalies", "outliers", "fill",
            "set", "change", "update", "replace", "make", "turn", "convert", "sort", "order", "top", "bottom", "rank", "it", "to", "with", "need", "of", "name", "breakdown", "if"
        }
        
        extracted_vals = []
        for token in doc_full:
            if token.text not in [str(c).lower() for c in found_cols] and token.text not in ignore_words:
                if token.pos_ in ["PROPN", "NOUN", "ADJ", "NUM"] and not token.like_num:
                    extracted_vals.append(token.text.capitalize())

        return {
            "columns": found_cols,
            "lemmas": lemma_terms,
            "target_column": target_column,
            "aggregation": aggregation,
            "aggregations": aggregations,
            "aggregation_column": aggregation_column,
            "group_by_column": group_by_column,
            "all_numeric_columns": all_numeric_columns,
            "numeric_columns": numeric_columns,
            "new_column_name": new_column_name,
            "column_formula": column_formula,
            "column_constant": column_constant,
            "row_values": row_values,
            "operation_clarification": operation_clarification,
            "conditions": conditions,
            "condition_column": condition_column,
            "condition_value": condition_value,
            "condition_is_exact": condition_is_exact,
            "clarification": clarification,
            "return_all_rows": return_all_rows,
            "operator": op,
            "numbers": all_numbers,
            "action_numbers": action_numbers,
            "cond_numbers": cond_numbers,
            "literal_values": extracted_vals,
            "replacement_from": replacement_from,
            "replacement_to": replacement_to,
            "raw_query": query_clean,
            "action_clause": action_clause,
            "cond_clause": cond_clause
        }


with open(__file__, "rb") as _source_file:
    SHEETLINGO_SOURCE_DIGEST = hashlib.sha256(_source_file.read()).hexdigest()
