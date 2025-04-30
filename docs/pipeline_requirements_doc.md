# Pipeline Requirements Document for "Talk to Your Data" Dashboard Data Pipeline

## 1. Overview

This document specifies the detailed requirements for the data pipeline that transforms the provided raw log data into a structured data mart for the “Talk to Your Data” Dashboard. The pipeline extracts, transforms, and loads (ETL) raw data into aggregate tables and intermediary datasets that feed individual dashboard components. It must cover data ingestion, normalization, session inference, metric aggregation, and text tokenization for the word cloud.

## 2. Raw Data Source Details

The raw data comes from a structured log file with the following columns (and descriptions):

- association_id  
  • Unique identifier for each LLM request.  
- timestamp  
  • Time when the request was sent to the LLM.
- user_email  
  • Email address of the user.
- data_source  
  • Origin of data (file, catalog, or database).
- query_type  
  • Type of event (e.g., “00_load_from_database_callback”, “02_rephrase”, “03_generate_code_database”, “03_generate_code_file”, “04_generate_run_charts_python_code”).
- user_msg  
  • Natural language query from the user.
- chat_id  
  • Identifier for the chat thread.
- chat_seq  
  • Numeric field indicating the order in the conversation (always multiples of 2).
- datasets_names  
  • Comma–separated dataset names selected by the user.
- enable_chart_generation  
  • Boolean flag (default True) indicating if chart generation is enabled.
- enable_business_insights  
  • Boolean flag (default True) indicating if business insights generation is enabled.
- error_message  
  • Error message if code generation fails.
- username  
  • Duplicate of user_email (result of table merge).
- visitTimestamp  
  • Timestamp when the user accessed the app.
- userId  
  • Account id (if missing, the user is in guest mode).
- userType  
  • “creator” (logged in) or “guest.”
- domain  
  • Domain of the user’s email (e.g., a.com, b.com).

## 3. Data Lineage and Transformation Steps

The following steps detail the full lineage from raw data to dashboard components:

### 3.1. Data Ingestion & Raw Storage

- **Ingestion Source:**  
  • Raw log file (CSV).

- **Initial Staging Table (raw_logs):**  
  • Columns as listed above.
  • Data validation at this stage:
    – Check that key fields (association_id, timestamp, chat_id, visitTimestamp) are not null.

### 3.2. Parsing and Normalization

- **Timestamp Parsing:**  
  • Convert ‘timestamp’ and ‘visitTimestamp’ into datetime objects,they are already at local (GMT+9) time zon.
  • Create additional columns:  
    – date (derived from timestamp)

- **Data Type Normalization:**  
  • Ensure booleans for enable_chart_generation and enable_business_insights are True/False.
  • Normalize string fields (user_email, username, domain).

- **User Identity Consolidation:**  
  • use user_email as unified user identifier.
  • Set userType flag accordingly.

- **Intermediary Cleaned Table (norm_logs):**  
  • Contains all normalized fields  
  • Additional columns:  
    – parsed_timestamp, parsed_visitTimestamp, date.

### 3.3. Session Inference

Since there is no logout event, the pipeline must infer sessions based on timestamps:

- **Session Grouping Logic:**  
  • Group records by unified user identifier (user_email).
  • Order events by timestamp.
  • A new session starts at the user’s visitTimestamp, or when the gap between consecutive events is greater than a threshold (e.g., 30 minutes).
  • Create fields:  
    – session_id (can be generated as a concatenation of user id and session start timestamp).
    – session_start (first event’s timestamp)  
    – session_end (timestamp of the last event in the session)  
    – session_duration = session_end − session_start.

- **Output Table (sessions):**  
  • Session-level records:
    – session_id, user_id, visitTimestamp, session_start, session_end, session_duration.
  • This table is used directly for the “Unique Sessions” KPI and “Session Duration Histogram.”

### 3.4. Chat & Query Flow Reconstruction

- **Chat Reconstruction:**  
  • Group norm_logs by chat_id and chat_seq.  
  • Order records by timestamp.
  • Identify a “conversation” flow:
    – the user message is in "user_msg" which should be the same for all rows with the same chat_id and chat_seq.
    – messages sent to llm (e.g., repeated “03_generate_code*” events) represent code-generation and retry cycles.
  • Create a breakdown log (chat_flow) where:
    - chat_no: chat_seq / 2, represent the order of the user chat in a thread
    – For each chat_id x chat_seq, record the start time (first record timestamp) and end time (last record timestamp).  
    – Count total events within the chat.
    – Derive a "retry_count_code_gen" field: For each chat_id x chat_seq group, this is the number of "03_generate_code_database" or "03_generate_code_file" events *minus 1*, representing the number of retries after the initial attempt. The value is capped at a minimum of 0. (`max(0, count_of_03_events - 1)`).
    – Derive a "retry_count_chart_gen" field: For each chat_id x chat_seq group, this is the number of "04_generate_run_charts_python_code" events *minus 1*, representing the number of retries after the initial attempt. The value is capped at a minimum of 0. (`max(0, count_of_04_events - 1)`).

- **Output Derived Table (chat_flows):**
  • Fields: chat_id, chat_seq (int), chat_no (int), user_id, number_of_events, retry_count_code_gen (int), retry_count_chart_gen (int), first_timestamp, last_timestamp, query_types_sequence (concatenated string).

### 3.5. KPI Aggregation & Metric Calculation

Using norm_logs, sessions, and chat_flows, aggregate metrics based on time dimensions (daily, weekly, monthly):

- **Unique Sessions:**  
  • Count distinct session_id by day/week/month from the sessions table.

- **Active Users:**  
  • Count distinct user identifiers (user_email) by period from norm_logs.

- **Total LLM Calls:**  
  • Count all records in norm_logs grouped by date (or other time granularity).
  • Also count for each query_type.

- **Chat Threads:**  
  • Count unique chat_id entries from chat_flows.

- **Chat User Messages:**  
  • Count unique [chat_id x chat_seq] entries from chat_flows.

- **Query Type Distribution:**
  • Group by query_type in norm_logs.
  • Produce counts per query_type within each time period.
  • Produce average retry counts (`avg_retry_code_gen`, `avg_retry_chart_gen`) per query_type within each time period. Note: Non-zero average retries will only appear for `03_generate_code*` and `04_generate_run_charts_python_code` types respectively; other types will have 0 for these averages.
  • Output table: `query_type_metrics` (fields: period, query_type, count, avg_retry_code_gen, avg_retry_chart_gen).

- **Feature Toggle Adoption:**  
  • For each period, count percent of events with enable_chart_generation=True and enable_business_insights=True.
  • Output table: feature_usage (fields: period, feature_name, true_count, false_count).

- **Error Analysis:**
  • In norm_logs, filter records where error_message is not null and not an empty string.
  • Group errors by error_message and count occurrences.
  # Note: Joining with chat_flows for context was deemed out of scope for the initial implementation but can be added later.
  • Output table: error_metrics (fields: period, error_category, count, sample_association_ids).

- **Dataset Usage Metrics:**  
  • Parse the datasets_names field (now it's a list stored as string) to get individual dataset names.
  • Count selections per dataset (and dataset combinations, max 2).
  • Output table: dataset_usage (fields: period, dataset_name, selection_count).

- **User Domain & Role Metrics:**  
  • Group by domain and userType to produce counts.
  • Output table: user_governance (fields: period, domain, userType, count).

### 3.6. Natural Language Processing for Word Cloud

- **Text Normalization:**  
  • For each record in norm_logs, take the user_msg field.
  • Normalize text through standard Unicode normalization (e.g., NFKC) and convert full-width to half-width where applicable.

- **Tokenization:**  
  • Use a Japanese morphological analyzer (fugashi).
  • Extract tokens with the following conditions:  
    – Include specific parts-of-speech (nouns, verbs, adjectives)
    – Exclude common Japanese stop-words (use an established stop-word list).

- **Frequency Distribution:**  
  • Aggregate tokens and compute frequency counts over the selected period.
  • Output table or JSON object: word_cloud_data (fields: token, frequency, period).
  • This output is fed to the dashboard’s word cloud component.

### 3.7. Data Storage & Query Interface

- **Final Aggregated Data Mart:**
  • Aggregated metric tables are stored as separate CSV files in the `output/` directory:
    - `daily_summary_metrics.csv`: Contains daily aggregated counts for unique sessions, active users, total LLM calls, chat threads, and chat user messages. (Result of merging individual daily metrics on 'date').
    - `daily_query_type_metrics.csv`: Daily counts and average retries per query type.
    - `daily_feature_usage.csv`: Daily counts of feature toggle usage (long format).
    - `daily_error_metrics.csv`: Daily counts of errors by category.
    - `daily_dataset_usage.csv`: Daily counts of dataset selections.
    - `daily_user_governance.csv`: Daily unique user counts by domain and user type.
    - `daily_word_cloud_data.csv`: Daily token frequencies for word cloud.
    - `sessions.csv`: Details of inferred user sessions.
    - `chat_flows.csv`: Details of reconstructed chat flows.

- **Schema Documentation:**
  • Provide a data dictionary detailing:
    – Table names.
    – Field definitions.
    – Data types.
    – Relationships between tables (e.g., sessions linked to norm_logs via user_email, chat_flows by chat_id and chat_seq, etc.).

### 3.8. Scheduling, Monitoring, and Automation

- **Batch Processing Schedule:**  
  • The pipeline is scheduled to run nightly (or at a configurable frequency) using orchestration tools which is out of scope.

- **Logging and Alerting:**  
  • Log every step in the pipeline. Record numbers of records processed, transformation timings, and errors.

- **Versioning and Lineage Tracking:**  
  • Track lineage from raw_logs → norm_logs → sessions, chat_flows, etc.  
  
## 4. Data Flow Diagram (Text Representation)

1. Raw Data (raw_logs)
   │
   ├── Validation & Ingestion → Raw Log Staging (raw_logs table)
   │
   └── Parsing & Normalization → Clean Data (norm_logs)
         │                          │
         ├─ Timestamp Parsing & Standardization  
         ├─ User Identity Consolidation  
         └─ Data Type Correction  
         
2. norm_logs
   │
   ├── Session Inference → Sessions Table (session_id, user_id, session_start, session_end, session_duration)
   │
   ├── Chat Flow Reconstruction → chat_flows (chat_id, chat_seq, retry_count_code_gen, retry_count_chart_gen, query_sequence)
   │
   ├── KPI Aggregation →  
   │       • query_type_metrics (grouped counts)  
   │       • feature_usage (counts by feature toggles)  
   │       • error_metrics (error counts and categories)  
   │       • dataset_usage (parsed from datasets_names)  
   │       • user_governance (grouped by domain and userType)
   │
   └── NLP Processing (user_msg Tokenization) → word_cloud_data (token, frequency)
   
3. Aggregated Tables in Data Mart  
   └─ Serve as source for Dashboard Queries

## 5. Tools & Implementation Details

- **Language & Libraries:**  
  • Python>3.11.  
  • Pandas for data manipulation.  
  • Dateutil (or built-in datetime) for time parsing.  
- **Japanese NLP:**  
  • fugashi.  
  • Use an established stop-word list for Japanese.
- **Orchestration:**  
  • Out of scope
- **Data Warehouse:**  
  • use csv for now

## 6. Acceptance Criteria

- The raw data is fully ingested with correct parsing and normalization.
- Sessions are accurately inferred, and each log record can be traced to a session.
- All dashboard metrics (sessions, active users, LLM calls, query type breakdown, feature toggles, error rates, dataset usage, and user governance) are correctly computed.
- The chat flow reconstruction correctly identifies and aggregates retry attempts.
- The word cloud tokenization accurately processes Japanese text and produces a reliable frequency table.
- All aggregated data is stored in a well-documented data mart that supports interactive querying by the dashboard.
- End-to-end lineage from raw_logs to each output table is clear and reproducible.

## 7. Future Considerations

- As additional fields (like explicit response timestamps or logout events) become available, extend the pipeline to incorporate these for richer performance metrics.
- Consider a near–real-time processing path for dashboards requiring a shorter refresh cycle.
- Implement more advanced NLP processing (e.g., topic modeling) when computational resources and data volumes allow.
