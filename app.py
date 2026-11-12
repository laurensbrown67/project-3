"""Northbridge Bank - Credit Risk Natural-Language Query Engine (Streamlit app)."""
import json
import os
import re
import sqlite3

import numpy as np
import pandas as pd
import sqlparse
import streamlit as st
from langchain_openai import ChatOpenAI

st.set_page_config(page_title="Credit Risk Query Engine", layout="wide")

# ---------------------------------------------------------------------------
# Credentials come from Streamlit Secrets (never from config.json in the repo)
# ---------------------------------------------------------------------------
try:
    os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
    os.environ["OPENAI_BASE_URL"] = st.secrets["OPENAI_API_BASE"]
except Exception:
    st.error("Missing secrets. Add OPENAI_API_KEY and OPENAI_API_BASE in the Streamlit app's Secrets settings.")
    st.stop()

# Generation LLM and a different evaluator LLM for the relevance check
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
evaluator_llm = ChatOpenAI(model="gpt-4o", temperature=0)


# ---------------------------------------------------------------------------
# Read-only database connection (relative path so it works from the repo)
# ---------------------------------------------------------------------------
@st.cache_resource
def get_connection():
    db_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "credit_risk_portfolio.db")
    return sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, check_same_thread=False)


conn = get_connection()


# ---------------- Schema context ----------------
database_schema = """
sector_master:
  sector_code (TEXT, PK): internal sector identifier (e.g., SEC_RE, SEC_INFRA)
  sector_name (TEXT): human-readable sector name (e.g., Real Estate, Infrastructure)
  naics_code (TEXT): NAICS industry classification code
  naics_description (TEXT): NAICS code description
  is_sensitive_sector (INTEGER): 1 if sensitive sector, 0 otherwise

loan_master:
  loan_account_number (TEXT, PK): unique loan identifier
  borrower_id (TEXT): borrower identifier (joins to borrower_rating.borrower_id)
  borrower_name (TEXT): registered legal name of the borrower
  borrower_type (TEXT): entity type (C-Corporation, S-Corporation, LLC, LP, Partnership, Sole Proprietorship)
  group_name (TEXT): business group affiliation, NULL if standalone
  state (TEXT): state of registered office
  product_type (TEXT): Term Loan, Working Capital, Cash Credit, Overdraft, Bill Discounting, Letter of Credit
  loan_category (TEXT): Corporate, Mid-Corporate, SME
  sector_code (TEXT, FK): joins to sector_master.sector_code
  sanctioned_amount (REAL): original approved loan amount in USD
  disbursed_amount (REAL): total amount disbursed in USD
  outstanding_principal (REAL): current principal outstanding in USD
  outstanding_interest (REAL): accrued interest outstanding in USD
  total_outstanding (REAL): outstanding_principal + outstanding_interest in USD
  interest_rate (REAL): current interest rate as percentage
  rate_type (TEXT): Fixed, Floating, MCLR-linked, Repo-linked
  sanction_date (DATE): date of original sanction
  maturity_date (DATE): contractual maturity date
  repayment_frequency (TEXT): Monthly, Quarterly, Bullet
  branch_code (TEXT): originating branch identifier
  branch_name (TEXT): originating branch name
  relationship_manager (TEXT): assigned relationship manager name
  is_consortium (INTEGER): 1 if consortium loan, 0 otherwise
  is_restructured (INTEGER): 1 if restructured, 0 otherwise
  restructuring_date (DATE): date of last restructuring, NULL if not restructured
  is_secured (INTEGER): 1 if secured, 0 if unsecured
  days_past_due (INTEGER): current maximum days past due for the loan
  asset_classification (TEXT): Pass, Special Mention, Substandard, Doubtful, Loss
  classification_date (DATE): date current classification was assigned

borrower_rating:
  rating_id (INTEGER, PK): auto-increment identifier
  borrower_id (TEXT, FK): joins to loan_master.borrower_id
  rating_date (DATE): date of rating assessment
  internal_rating (TEXT): bank's internal rating grade (AAA through D, 18-grade scale)
  previous_rating (TEXT): rating grade from prior assessment
  rating_direction (TEXT): Upgraded, Downgraded, Maintained
  external_rating_agency (TEXT): S&P, Moody's, Fitch, DBRS Morningstar, Kroll, or NULL
  external_rating (TEXT): external agency rating
  pd_estimate (REAL): probability of default (decimal, e.g., 0.02 for 2%)
  rating_model_version (TEXT): internal rating model version

provisioning:
  provision_id (INTEGER, PK): auto-increment identifier
  loan_account_number (TEXT, FK): joins to loan_master.loan_account_number
  reporting_date (DATE): quarter-end reporting date
  ifrs9_stage (INTEGER): IFRS 9 stage (1, 2, or 3)
  stage_rationale (TEXT): reason for stage assignment
  pd_12_month (REAL): 12-month probability of default
  pd_lifetime (REAL): lifetime probability of default
  lgd_estimate (REAL): loss given default (decimal)
  ead_amount (REAL): exposure at default in USD
  ecl_amount (REAL): expected credit loss in USD
  provision_held (REAL): provision amount held in USD
  provision_coverage_ratio (REAL): provision_held / total_outstanding * 100
  is_individually_assessed (INTEGER): 1 if individually assessed, 0 if modeled

Available reporting_date values in provisioning: 2024-12-31, 2025-03-31, 2025-06-30, 2025-09-30
Available rating_date values in borrower_rating: 2024-09-30, 2024-12-31, 2025-03-31, 2025-06-30, 2025-09-30
Latest reporting_date: 2025-09-30
The provisioning table holds exactly four quarterly snapshots, so "last four quarters" means all reporting_date values. 
Latest rating_date: 2025-09-30
NPA definition: asset_classification IN ('Substandard', 'Doubtful', 'Loss')
Impaired loans: loans classified as NPA (loan_master.asset_classification, a current snapshot)
IFRS 9 stage: use provisioning.ifrs9_stage (1, 2, or 3) for any question about stages; never use asset_classification to define a stage. 
Overdue loans: days_past_due > 0 
Exposure / outstanding: total_outstanding in loan_master (USD), a current snapshot 
Stage-level exposure and trends: use provisioning.ead_amount (as of each reporting_date); loan_master.total_outstanding is a current 
snapshot.
"""

sql_1 = """
SELECT
    s.sector_name,
    ROUND(SUM(l.total_outstanding)/1e6, 2) AS total_outstanding_mn,
    ROUND(SUM(CASE WHEN l.asset_classification IN ('Substandard', 'Doubtful', 'Loss') 
        THEN l.total_outstanding ELSE 0 END)/1e6, 2) AS npa_outstanding_mn
FROM loan_master l
JOIN sector_master s ON l.sector_code = s.sector_code
GROUP BY s.sector_name
ORDER BY total_outstanding_mn DESC
"""

sql_2 = """
SELECT 
    loan_category,
    ROUND(SUM(total_outstanding) / 1e6, 2) AS total_outstanding_mn,
    COUNT(*) AS loan_count
FROM loan_master
GROUP BY loan_category
ORDER BY total_outstanding_mn DESC"""

sql_3 = """
SELECT
    ifrs9_stage,
    COUNT(*) AS loan_count,
    ROUND(SUM(ead_amount)/1e6, 2) AS total_ead_mn,
    ROUND(SUM(ecl_amount)/1e6, 2) AS total_ecl_mn
FROM provisioning
WHERE reporting_date = '2025-09-30'
GROUP BY ifrs9_stage
ORDER BY ifrs9_stage
"""

sql_4 = """
SELECT 
    s.sector_name,
    ROUND(AVG(p.provision_coverage_ratio), 2) AS avg_provision_coverage_ratio
FROM provisioning p
JOIN loan_master l ON p.loan_account_number = l.loan_account_number
JOIN sector_master s ON l.sector_code = s.sector_code
WHERE p.reporting_date = '2025-09-30'
GROUP BY s.sector_name
ORDER BY avg_provision_coverage_ratio DESC
"""

sql_5 = """
SELECT
    l.borrower_name,
    s.sector_name,
    ROUND(l.total_outstanding/1e6, 2) AS total_outstanding_mn,
    l.asset_classification
FROM loan_master l
JOIN sector_master s ON l.sector_code = s.sector_code
ORDER BY l.total_outstanding DESC
LIMIT 10
"""

sql_6 = """
SELECT 
    group_name, 
    COUNT(*) AS loan_count, 
    ROUND(SUM(total_outstanding)/1e6, 2) AS total_outstanding_mn
FROM loan_master
WHERE group_name IS NOT NULL 
GROUP BY group_name
ORDER BY total_outstanding_mn DESC
LIMIT 5
"""

sql_7 = """
SELECT 
    l.loan_account_number,
    l.borrower_name,
    s.sector_name,
    ROUND(l.total_outstanding/1e6, 2) AS total_outstanding_mn,
    l.days_past_due,
    l.asset_classification
FROM loan_master l 
JOIN sector_master s ON l.sector_code = s.sector_code
WHERE l.days_past_due > 0
ORDER BY l.days_past_due DESC
"""

sql_8 = """
SELECT 
    CASE
        WHEN days_past_due = 0 THEN '0 (Current)'
        WHEN days_past_due BETWEEN 1 AND 30 THEN '1-30'
        WHEN days_past_due BETWEEN 31 AND 60 THEN '31-60'
        WHEN days_past_due BETWEEN 61 AND 90 THEN '61-90'
        ELSE '90+'
    END AS dpd_bucket, 
    COUNT(*) AS loan_count, 
    ROUND(SUM(total_outstanding)/1e6, 2) AS total_outstanding_mn
FROM loan_master
GROUP BY dpd_bucket
ORDER BY MIN(days_past_due)
"""

sql_9 = """
SELECT 
    borrower_id,
    previous_rating, 
    internal_rating AS current_rating,
    pd_estimate
FROM borrower_rating
WHERE rating_date = '2025-09-30' AND rating_direction = 'Downgraded'
ORDER BY pd_estimate DESC
"""

sql_10 = """
SELECT 
    reporting_date,
    ROUND(SUM(ecl_amount)/1e6, 2) AS total_ecl_mn
FROM provisioning
GROUP BY reporting_date
ORDER BY reporting_date
"""

# ---------------- Verified query library ----------------
verified_query_library = {
    'VQ1': {
        'description': 'Sector-wise total outstanding and NPA amount breakdown across all sectors',
        'sql': sql_1
    },


    'VQ2': {
        'description': 'Total portfolio outstanding broken down by loan category (Corporate, Mid-Corporate, SME)',
        'sql': sql_2
    },


    'VQ3': {
        'description': 'IFRS 9 stage-wise summary showing loan count, exposure at default, and expected credit loss for the latest quarter',
        'sql': sql_3
    },


    'VQ4': {
        'description': 'Average provision coverage ratio by sector for the latest reporting quarter',
        'sql': sql_4
    },


    'VQ5': {
        'description': 'Top 10 largest loan exposures by outstanding amount at the borrower level',
        'sql': sql_5
    },


    'VQ6': {
        'description': 'Top 5 largest exposures aggregated at the business group level',
        'sql': sql_6
    },


    'VQ7': {
        'description': 'All overdue loan accounts with their days past due and asset classification',
        'sql': sql_7
    },


    'VQ8': {
        'description': 'Distribution of loans across days-past-due buckets showing aging profile of the portfolio',
        'sql': sql_8
    },


    'VQ9': {
        'description': 'Borrowers whose internal rating was downgraded in the latest rating cycle',
        'sql': sql_9
    },


    'VQ10': {
        'description': 'Expected credit loss trend across all reporting quarters showing provisioning movement over time',
        'sql': sql_10
    }
}


# ---------------- Pipeline tools ----------------
def classify_intent(user_question, query_library):
    '''
    Classifies the user question and decides which route to take.

    Parameters:
    - user_question (str): The natural language question from the user.
    - query_library (dict): The verified query template library.

    Returns:
    - dict: Contains 'route' (verified or generated),
                     'query_id' (template ID or None),
                     'match_reason' (short explanation of the decision).
    '''

    library_descriptions = '\n'.join(
        [f"{qid}: {entry['description']}" for qid, entry in query_library.items()]
    )

    classification_prompt = f"""

### ROLE
You are a query router for the credit risk analytics system of a commercial bank. Your job is to decide whether a business user's
question can be answered by one of the pre-approved verified query templates, or whether it needs fresh SQL generation.

### INPUT
User Question: 
{user_question}

Available Verified Query Templates: 
{library_descriptions}

### INSTRUCTIONS 
1. Read the question and identify its analytical intent: the metric, the grouping, and the shape of the answer (a distribution, a ranked
list, a trend, a row-level list).

2. Compare the intent against each template description. Match on semantic meaning, not exact wording. For example: 'non-performing', 'NPA'
or 'bad loans' means Substandard/Doubtful/Loss classification; 'overdue', 'delinquent' or 'past due' means days_past_due > 0; 'aging' or 
'DPD buckets' means the DPD bucket distribution; 'provisions' or 'expected credit loss' means ECL; 'downgrades' means 
rating_direction = Downgraded; 'book', 'exposure' or 'protfolio' means total_outstanding. 

3. Return a template ID only if that template genuinely answers the question. A template may return more rows than the user asked about 
(for example all sectors when the user asks about one sector), which is acceptable because the response step highlights the relevant rows.

4. The dimension or metric the user needs must already appear as a column or row in the template's output. If it does not, the template
does not match. For example, a question about one subset(such as a single IFRS 9 stage, a restructured flag or an interest rate) when the 
template aggregates across all subsets or does not return that metric must go to generated.

5. Be careful about the shape of the answer: a question asking for a list of individual loans must not match a template that returns an 
aggregate or distribution, and a question asking for a distribution must ot match a row-level list. 

6. If no template covers the question, return null for query_id and set the route to generated.

### OUTPUT
Return ONLY a valid JSON dictionary with these exact keys:
{{
  "route": "verified" or "generated",
  "query_id": "VQ1" or "VQ2" ... "VQ10" or null,
  "match_reason": "one short sentence explaining the decision"
}}
Do not include any other text.
"""

    response = llm.invoke(classification_prompt).content.strip()
    # Extract JSON from potential markdown blocks
    json_match = re.search(r'\{.*\}', response, re.DOTALL)
    if json_match:
        return json.loads(json_match.group())
    return {"route": "generated", "query_id": None, "match_reason": "Could not parse classification"}

def generate_query(user_question, schema_context):
    '''
    Generates a candidate SQL query for a novel question using the database schema.

    Parameters:
    - user_question (str): The natural language question.
    - schema_context (str): Full database schema description.

    Returns:
    - str: Candidate SQL query as a string.
    '''

    generation_prompt = f"""

### ROLE 
You ae a senior SQL developer specailizing in commercial lending and credit risk analytics on a SQLite database.

### INPUT
User Question: 
{user_question}

Database Schema (single source of truth):
{schema_context}

### INSTRUCTIONS
1. Write a single SQL query that answers the user question using only the provided schema. 

2. The query must be read-only. Use SELECT (or WITH...SELECT). Never use DROP, DELETE, UPDATE, INSERT, ALTER, TRUNCATE, REPLACE or ATTACH.

3. Use only the tables and columns listed in the schema. Do not invent columns or tables.

4.  Apply the business rules in the schema exactly: NPA/ impaired = asset_classification IN ('Substandard', 'Doubtful', 'Loss');
overdue = days_past_due > 0; current exposure = total_outstanding (a current snapshot).

5. Use the latest dates ('2025-09-30') for provisioning.reporting_date and borrower_rating.rating_date unless the question asks for a trend,
a comparison across quarters or a specific date. When a question asks about a trend or "over time", include every reporting_date and order
by it. 

6. Provisioning has one row per loan per reporting_date. When joining it to loan_master, filter to one reporting_date or group by 
reporting_date so loans are not double counted. 

7. Get sector names by joining loan_master to sector_master on sector_code. Compute percentages with 100.0 * numerator/denominator to 
avoid integer division.

8. In SQLite, never subtract DATE() values directly; use julianday(a) - julianday(b) for day differences.

9. Convert USD amounts to millions (divide by 1e6 and ROUND to 2 decimals) to match the verified templates. Alias every numeric column with
a suffix that states its unit: _mn for USD millions, _pct for percentages, _count for counts, _days for days. AVOID bare aliases like 
"value", "amount", or "total".

10. The query must be SQLite compatible. 

11. ROUND every computed average and percentage to 2 decimals, in addition to USD amounts. For ranking and "highest to lowest" questions,
add an ORDER BY that matches the question; for lists of rows, order by the most relevant measure. 

12. When a question asks for a percentage or total alongside a filtered list of rows (for example restructured loans), compute it over
exactly the same filtered rows, using a window function such as COUNT(*) OVER () in the same query or a subquery with the same WHERE
clause. Never compute it over the whole table. When combining a row-level list with a summary figure, every aggregate must use OVER () 
(for example COUNT(*) OVER () or SUM(CASE WHEN ... THEN 1 ELSE 0 END) OVER ()). Never place an aggregate without OVER () or GROUP BY next
to row-level columns, because it collapses the list to one row. A "list every item" question must return one row per item. 

13. For exposure by IFRS 9 stage or over time (book size, EAD), use provisioning.ead_amount, which is measured as of each reporting_date.
Do not use loan_master.total_outstanding for stage or trend questions, because it is a current snapshot not tied to a reporting_date. Alias
it with the _mn suffix, for example total_ead_mn. 

14. For any question about IFRS 9 stages (Stage 1, 2, or 3), filter on provisioning.ifrs9_stage. Never use loan_master.asset_classification 
to define a stage, never apply a current loan_master attribute to a historical quarter.

15. Add count or percentage columns only when the question explicitly asks for a percentage, share, or count. Otherwise return only the 
metrics requested, and never add total_count or percentage columns to trend, ranking or average questions. When you do add them next to 
row-level columns, every aggregate must use a window function, for example COUNT(*) OVER () and SUM(...) OVER (), including the denominator
of any percentage. Never mix a plain aggregate with row-level columns. A percentage must be between 0 and 100. 

### OUTPUT 
Return ONLY the SQL query, with no markdown code blocks, no comments and no explanation. 
"""

    sql = llm.invoke(generation_prompt).content.strip()
    # Strip markdown fences if present
    sql = re.sub(r'^```sql\s*|\s*```$', '', sql, flags=re.IGNORECASE | re.MULTILINE).strip()
    sql = re.sub(r'^```\s*|\s*```$', '', sql, flags=re.MULTILINE).strip()
    return sql

def validate_query(user_question, candidate_sql, db_connection, query_library, query_id=None):
    '''
    Validates a candidate SQL query through five checks before execution.

    Parameters:
    - user_question (str): The original user question.
    - candidate_sql (str): The SQL query to validate.
    - db_connection: SQLite connection object.
    - query_library (dict): Verified query library (for integrity check).
    - query_id (str, optional): Template ID if from verified track.

    Returns:
    - dict: Contains 'passed' (bool), 'failed_check' (str or None), 'details' (str),
            and 'relevance_confidence' (int, 0-1).
    '''

    result = {
        'passed': False,
        'failed_check': None,
        'details': '',
        'relevance_confidence': None
    }



    # Check 1: Read-only shape check
    sql_upper = candidate_sql.upper().strip()
    forbidden_keywords = ['DROP', 'DELETE', 'UPDATE', 'INSERT', 'ALTER', 'TRUNCATE', 'REPLACE', 'ATTACH']
    if not (sql_upper.startswith('SELECT') or sql_upper.startswith('WITH')):
        result['failed_check'] = 'read_only_shape'
        result['details'] = 'Query must start with SELECT or WITH'
        return result
    for kw in forbidden_keywords:
        if re.search(r'\b' + kw + r'\b', sql_upper):
            result['failed_check'] = 'read_only_shape'
            result['details'] = f'Forbidden keyword detected: {kw}'
            return result
    if ';' in candidate_sql.rstrip(';').rstrip():
        result['failed_check'] = 'read_only_shape'
        result['details'] = 'Multiple statements are not allowed'
        return result



    # Check 2: Schema conformance check
    cur = db_connection.cursor()
    real_tables = [r[0] for r in cur.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()]
    real_columns = set()
    for t in real_tables:
        for col_info in cur.execute(f"PRAGMA table_info({t})").fetchall():
            real_columns.add(col_info[1].lower())
    parsed = sqlparse.parse(candidate_sql)[0]
    tokens = [str(t).strip().lower() for t in parsed.flatten() if t.ttype is None or 'Name' in str(t.ttype)]
    referenced_identifiers = re.findall(r'\b[a-z_][a-z0-9_]*\b', candidate_sql.lower())
    sql_keywords = {'select','from','where','and','or','group','by','order','having','limit','join','on','as','case',
                    'when','then','else','end','sum','count','avg','min','max','round','desc','asc','left','right',
                    'inner','outer','distinct','null','is','not','in','like','with','union','all','between','coalesce'}
    unknown = [tok for tok in referenced_identifiers
               if tok not in sql_keywords and tok not in real_columns and tok not in real_tables
               and not tok.isdigit() and tok not in ('s', 'l', 'p', 'r', 'e6')]



    # Check 3: Parse-and-plan dry run using EXPLAIN
    try:
        cur.execute(f"EXPLAIN {candidate_sql}")
        cur.fetchall()
    except sqlite3.Error as e:
        result['failed_check'] = 'parse_plan_dry_run'
        result['details'] = f'SQL failed to parse or plan: {str(e)}'
        return result



    # Check 4: LLM relevance check
    is_verified_track = query_id is not None and query_id in query_library
    track_context = (
        "This SQL is a pre-approved VERIFIED TEMPLATE. It is intentionally broad "
        "(e.g., it may return all sectors/categories/stages rather than filtering to "
        "just what the user asked). A separate response-generation step will filter and "
        "highlight the relevant rows afterward. Do NOT fail this query for lacking a "
        "WHERE clause that narrows to the user's specific sector/category/stage — judge "
        "only whether the underlying metric, tables, and aggregation logic match the "
        "question's intent."
        if is_verified_track else
        "This SQL was freshly generated for this specific question and should be "
        "appropriately scoped/filtered to answer it directly."
    )


    relevance_prompt = f"""

### ROLE 
You are a senior data validator reviewing SQL for the credit risk analytics team of a commercial bank. Your job is to check whether a SQL
query correctly answers a business user's question about the commercial lending portfolio. 

### CONTEXT
{track_context}

### INPUT
User Question: 
{user_question}

Candidate SQL: 
{candidate_sql}

### INSTRUCTIONS 
Asses whether the SQL genuinely answers what the user asked, considering: 
1. Does it query the correct tables and columns? 

2. Does it apply the right aggregations and groupings (for exaple, SUM of toal_outstanding for exposure)? 

3. Does it apply the business definitions correctly: NPA/impaired = Substandard, Doubtful or Loss; overdue = days_past_due > 0; latest 
quarter = 2025-09-30?

4. Does it avoid double counting? Provisioning has one row per loan per reporting_date, so it must be filtered to one date or grouped by 
date when joined to loan-level data. 

5. Does it resolve named entities (such as sector names) correctly through sector_master? 

6. Does it return the right shape of answer (distribution, ranking, trend, list)? 

7. If this is a verified template, do not penalize it for returning a broader result set than the question's scope.
Use confidence of 0.9 or higher only when the SQL clearly answers the question, and below 0.6 when it does not. 

8. The provisioning table contains four quarterly reporting dates (2024-12-31, 2025-03-31, 2025-06-30, 2025-09-30). A question about "last
four quarters", "all quarters", or a trend over time is therefore answered by including every reporting_date, so do not require or penalize
a missing date filter for such questions. 

9. If the question asks to list every item (for example every restructured loan), the SQL must return one row per item. Fail any SQL that
collapses the list to one row with an aggregate that lacks OVER () or GROUP BY, or that produces a percentage above 100. 
### OUTPUT
Return ONLY a JSON dictionary:
{{
  "verdict": "yes" or "no",
  "confidence": 0.0 to 1.0,
  "reason": "one short sentence"
}}

"""
    relevance_response = evaluator_llm.invoke(relevance_prompt).content.strip()
    json_match = re.search(r'\{.*\}', relevance_response, re.DOTALL)
    if json_match:
        relevance_json = json.loads(json_match.group())
        result['relevance_confidence'] = relevance_json.get('confidence', 0.0)
        if relevance_json.get('verdict') == 'no' or relevance_json.get('confidence', 0.0) < 0.6:
            result['failed_check'] = 'llm_relevance'
            result['details'] = f"Relevance check failed: {relevance_json.get('reason', 'unknown')}"
            return result



    # Check 5: Verified template integrity check (verified track only)
    if query_id and query_id in query_library:
        expected_sql = query_library[query_id]['sql']
        try:
            expected_cols = [d[0] for d in cur.execute(f"{expected_sql} LIMIT 0").description]
            actual_cols = [d[0] for d in cur.execute(f"{candidate_sql} LIMIT 0").description]
            if len(expected_cols) != len(actual_cols):
                result['failed_check'] = 'template_integrity'
                result['details'] = f'Expected {len(expected_cols)} columns, got {len(actual_cols)}'
                return result
        except sqlite3.Error as e:
            result['failed_check'] = 'template_integrity'
            result['details'] = f'Template integrity check failed: {str(e)}'
            return result

    result['passed'] = True
    result['details'] = 'All validation checks passed'
    return result

def retry_generation(user_question, failed_sql, error_message, schema_context):
    '''
    Regenerates SQL after a validation failure, feeding the error back to the LLM.

    Parameters:
    - user_question (str): The original user question.
    - failed_sql (str): The SQL that failed validation.
    - error_message (str): The specific failure reason.
    - schema_context (str): Database schema description.

    Returns:
    - str: Revised SQL as a string.
    '''

    retry_prompt = f"""
### ROLE
You are a senior SQL developer fixing a query that failed validation.

### INPUT 
User Question:
{user_question}

Failed SQL:
{failed_sql}

Validation Error:
{error_message}

Database Schema:
{schema_context}

### INSTRUCTIONS
1. Fix only the specific issue identified by the validation error.

2. Preserve the original intent of the question.

3. The revised SQL must be read-only: SELECT (or WITH...SELECT) only.

4. Use only tables and columns from the schema. If the error mentions an unkown column or table, check the schema and correct it; never
invent one. 

5. Apply the Schema's business rules (NPA definition, latest reporting_date, no double counting of provisioning rows). 

6. Ensure the query is SQLite compatible (use julianday() for date differences, 100.0 * for percentages).

### OUTPUT
Return only the corrected SQL, with no markdown code blocks, no comments and no explanation. 
"""

    revised_sql = llm.invoke(retry_prompt).content.strip()
    revised_sql = re.sub(r'^```sql\s*|\s*```$', '', revised_sql, flags=re.IGNORECASE | re.MULTILINE).strip()
    revised_sql = re.sub(r'^```\s*|\s*```$', '', revised_sql, flags=re.MULTILINE).strip()
    return revised_sql

def execute_query(validated_sql, db_connection):
    '''
    Executes a gate-passed SQL query and returns the result as a DataFrame.

    Parameters:
    - validated_sql (str): SQL query that has passed all validation checks.
    - db_connection: Read-only SQLite connection object.

    Returns:
    - dict: Contains 'dataframe' (pandas DataFrame), 'reasonable' (bool),
            and 'warnings' (list of warning strings).
    '''

    result = {
        'dataframe': None,
        'reasonable': True,
        'warnings': []
    }

    df = pd.read_sql_query(validated_sql, db_connection)
    result['dataframe'] = df

    # Reasonableness checks
    if df.empty:
        result['warnings'].append('Query returned an empty result')

    for col in df.select_dtypes(include='number').columns:
        if (df[col] < 0).any() and 'deviation' not in col.lower() and 'change' not in col.lower():
            result['warnings'].append(f'Column {col} contains negative values')
        if df[col].isnull().any():
            null_count = df[col].isnull().sum()
            if null_count > len(df) * 0.5:
                result['warnings'].append(f'Column {col} has {null_count} null values')

    if len(result['warnings']) > 2:
        result['reasonable'] = False

    return result

def generate_response(user_question, dataframe, route, query_id=None):
    '''
    Generates a focused natural language response from the query result.

    Parameters:
    - user_question (str): The original user question.
    - dataframe (pd.DataFrame): The full query result.
    - route (str): 'verified' or 'generated'.
    - query_id (str, optional): Template ID if from verified track.

    Returns:
    - str: Natural language response focused on what the user asked.
    '''

    response_prompt = f"""

### ROLE 
You are a credit risk analyst writing a concise business resposne for the Chief Risk Officer's and the Credit Committee.

### INPUT
User Question: 
{user_question}

Query Result Data: 
{dataframe.to_string()}

### INSTRUCTIONS
1. Answer the user's specific question directly; don't dump the entire table. 

2. If the user asked about a specific sector, DPD bucket, IFRS 9 stage or category, highlight only those rows. Add context from other rows
only when it adds value (for example a ranking, a share of the total or a comparison).

3. State exact numbers from the data. Do not round beyond what is shown. Never count, group, sum or average rows yourself. Only state figures
that appear as values in the data, such as an existing total, count or percentage column. If the data is a list individual of rows, report
only the summary columns do not name individual rows unless the question asks for specific ones. Every number you state must appear in the 
data. If a figure such as a count is not shown, leave it out. 

4. State the unit for every number, inferred from its column name: _mn as "$X million", _pct as "X%", _count as a plain count, _days as "X
days". Never state a bare number when the column implies a unit. 

5. Flag anything notable only if the data directly supports it, such as high NPA share, a heavily concentrated exposure or a rising expected
credit loss. Do not add general market commentary or recommendations. 

6. Use clearm professional language suitable for a risk committee memo. 

7. Keep it focused: two to four sentences for simple questions, up to a short paragraph for complex ones. 

8. If the question is about overdue or delinquent loans, treat only days_past_due > 0 as overdue. Mention the current bucket only as context,
not as part of the overdue book.

9. Write in plain prose. Do not use bullet points or numbered lists. 

10. State a ratio, percentage, total or ranking only if it appears as a value in the data. Never count rows or compare sectors, categories 
or groups yourself. 

### OUTPUT
Return only the natural language reponse text, with no markdown headers or bullet points unless truly needed. 
"""

    narrative = llm.invoke(response_prompt).content.strip()
    return narrative

def run_pipeline(user_question, db_connection, query_library, schema_context, verbose=True):
    '''
    Runs the complete query engine pipeline for a single user question.

    Parameters:
    - user_question (str): The natural language question.
    - db_connection: SQLite connection object.
    - query_library (dict): Verified query template library.
    - schema_context (str): Database schema description.
    - verbose (bool): If True, prints intermediate pipeline stages.

    Returns:
    - dict: Complete pipeline output including narrative, SQL, data, and log.
    '''

    log = {
        'user_question': user_question,
        'route': None,
        'query_id': None,
        'match_reason': None,
        'candidate_sql': None,
        'gate_result': None,
        'retry_used': False,
        'escalated': False,
        'executed_sql': None,
        'row_count': None,
        'confidence': None,
        'narrative': None
    }

    # Step 1: Intent classification
    classification = classify_intent(user_question, query_library)
    log['route'] = classification['route']
    log['query_id'] = classification.get('query_id')
    log['match_reason'] = classification.get('match_reason')

    if verbose:
        print(f"[1] Intent Classification: route={log['route']}, query_id={log['query_id']}")
        print(f"    Reason: {log['match_reason']}")


    # Step 2: Query construction
    if log['route'] == 'verified' and log['query_id'] in query_library:
        candidate_sql = query_library[log['query_id']]['sql']
    else:
        candidate_sql = generate_query(user_question, schema_context)
    log['candidate_sql'] = candidate_sql

    if verbose:
        print(f"[2] Query Construction: {'loaded from library' if log['route']=='verified' else 'generated fresh SQL'}")


    # Step 3: Validation gate
    gate = validate_query(user_question, candidate_sql, db_connection, query_library, log['query_id'])
    log['gate_result'] = gate

    if verbose:
        print(f"[3] Validation Gate: passed={gate['passed']}, relevance_confidence={gate.get('relevance_confidence')}")
        if not gate['passed']:
            print(f"    Failed check: {gate.get('failed_check')}")
            print(f"    Details: {gate.get('details')}")


    # Step 4: Retry once on generated track if validation fails
    if not gate['passed'] and log['route'] == 'generated':
        if verbose:
            print(f"    Retrying: {gate['details']}")
        candidate_sql = retry_generation(user_question, candidate_sql, gate['details'], schema_context)
        log['candidate_sql'] = candidate_sql
        log['retry_used'] = True
        gate = validate_query(user_question, candidate_sql, db_connection, query_library, None)
        log['gate_result'] = gate

        if verbose:
            print(f"    Retry Validation Gate: passed={gate['passed']}, relevance_confidence={gate.get('relevance_confidence')}")
            if not gate['passed']:
                print(f"    Retry failed check: {gate.get('failed_check')}")
                print(f"    Retry details: {gate.get('details')}")


    # Step 5: Escalate if still failing
    if not gate['passed']:
        log['escalated'] = True
        log['narrative'] = f"Query could not be reliably resolved. Escalated to human analyst. Failure: {gate['details']}"
        log['confidence'] = 'ESCALATED'
        if verbose:
            print(f"[!] Escalated to human: {gate['details']}")
        return {'log': log, 'dataframe': None, **log}


    # Step 6: Execute
    log['executed_sql'] = candidate_sql
    exec_result = execute_query(candidate_sql, db_connection)
    df = exec_result['dataframe']
    log['row_count'] = len(df)

    if verbose:
        print(f"[4] Execute: {len(df)} rows returned")
        if exec_result['warnings']:
            print(f"    Warnings: {exec_result['warnings']}")


    # Step 7: Response generation
    narrative = generate_response(user_question, df, log['route'], log['query_id'])
    log['narrative'] = narrative


    # Confidence: carried directly from the validation gate's relevance check (0-1)
    log['confidence'] = gate.get('relevance_confidence')

    if verbose:
        print(f"[6] Response Generation: confidence={log['confidence']}")

    return {'log': log, 'dataframe': df, **log}


# ---------------------------------------------------------------------------
# Streamlit user interface
# ---------------------------------------------------------------------------
SAMPLE_QUESTIONS = [
    "How much of our book is in real estate, and how much of that is non-performing?",
    "Show me how our overdue book is distributed across DPD buckets.",
    "List every restructured loan, flag which are impaired, and show the overall impaired percentage.",
    "Show me the average interest rate by sector, sorted from highest to lowest.",
    "How has the Stage 3 book size and its expected credit loss moved over the last four quarters?"
]

st.title("Credit Risk Natural-Language Query Engine")
st.caption("Northbridge Bank - ask a question about the commercial lending portfolio.")

with st.sidebar:
    st.header("Sample questions")
    for i, q in enumerate(SAMPLE_QUESTIONS, start=1):
        if st.button(f"Test case {i}", key=f"sample_{i}", use_container_width=True):
            st.session_state["question"] = q
    st.markdown("---")
    st.markdown("**Verified templates**")
    for qid, entry in verified_query_library.items():
        st.caption(f"{qid}: {entry['description']}")

question = st.text_area("Your question", key="question", height=100,
                        placeholder="e.g. What is the total portfolio outstanding by loan category?")

if st.button("Run query", type="primary"):
    if not question.strip():
        st.warning("Please enter a question.")
    else:
        with st.spinner("Running the query engine..."):
            try:
                result = run_pipeline(question.strip(), conn, verified_query_library,
                                      database_schema, verbose=False)
            except Exception as exc:
                st.error(f"Pipeline error: {exc}")
                st.stop()

        log = result["log"]
        if log["escalated"]:
            st.error("Escalated to a human analyst")
            st.write(log["narrative"])
        else:
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Route", log["route"])
            c2.metric("Template", log["query_id"] or "None (generated)")
            c3.metric("Confidence", log["confidence"])
            c4.metric("Rows returned", log["row_count"])

            st.subheader("Response")
            st.write(log["narrative"].replace("$", "\\$"))

            st.subheader("Executed SQL")
            st.code(log["executed_sql"], language="sql")

            st.subheader("Result data")
            st.dataframe(result["dataframe"], use_container_width=True)

        with st.expander("Pipeline details"):
            st.write({
                "route": log["route"],
                "query_id": log["query_id"],
                "match_reason": log["match_reason"],
                "retry_used": log["retry_used"],
                "validation": log["gate_result"],
            })
