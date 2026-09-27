# SheetLingo

SheetLingo lets people inspect and clean CSV or Excel data with plain-language commands. It combines dataset profiling, natural-language query handling, previews for data changes, version history, and a DuckDB SQL workspace.

## Run locally

Use Python 3.12, then run these commands from the project folder:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m spacy download en_core_web_sm
python -m streamlit run app.py
```

The first run may download the spaCy English model and the Sentence Transformers model used for intent matching. Keep an internet connection available for that first run.

## Example commands

- `show John`
- `show rows where Department is Sales`
- `show active employees older than 40`
- `show employees older than forty and salary over 80k`
- `replace Sales with Sells in Department`
- `show all employees`

Edits appear as an uncommitted preview. Review the affected rows and counts, then commit or discard the change.

## How command handling works

SheetLingo uses spaCy token and part-of-speech processing, sentence embeddings to choose an intent, fuzzy column matching, a small set of column synonyms, and matching against values in the active dataset. It supports common numeric comparisons and combines clearly parsed filter conditions with AND. When it cannot safely identify a filter or edit, it asks for a clearer request rather than applying a guessed change. OR filters, broad open-ended conversation, and non-English commands are not currently supported.

The embedding model helps select an operation; it does not interpret arbitrary language like a general-purpose conversational model. Review every proposed edit before committing it.
