
# Data Lineage & Pipeline – Detailed Processing Steps

This document describes how to transform the raw log data into intermediate datasets and final output datasets that power the “Talk to Your Data” dashboard. Each stage, its operations, its output, and the corresponding dashboard element(s) are described in detail.

---

## 1. Raw Data Ingestion & Normalization

### Input  
Raw log data with the following columns:  
• association_id  
• timestamp  
• user_email  
• data_source  
• query_type  
• user_msg  
• chat_id  
• chat_seq  
• datasets_names  
• enable_chart_generation  
• enable_business_insights  
• error_message  
• username  
• visitTimestamp  
• userId  
• userType  
• domain  

### Operations  
- Convert time-related fields (`timestamp`, `visitTimestamp`) into datetime objects (normalize timezone if needed).  
- Clean and standardize string fields (trim whitespace, lowercase emails, etc.).  
- Validate that each `association_id` is unique.  
- Unify user identity by mapping and/or merging `user_email`, `username`, and `userId` into a consistent identifier. 

### Output Dataset  
**Dataset Name:** Raw_Cleaned_Log  
**Purpose:**  
This dataset is the “golden copy” for subsequent transformations. All downstream pipelines use this cleaned version to ensure consistency.

---

## 2. Session Inference and Aggregation

### Input  
Raw_Cleaned_Log  

### Operations  
- **Grouping by User:**  
  Group records by user identifier (using user_email or userId) sorted by timestamp.  
- **Session Identification:**  
  • Use the `visitTimestamp` as the session start.  
  • Identify the session end by taking the last `timestamp` (LLM call) within that session.  
  • Optionally, if the time gap between consecutive requests is more than a threshold (e.g., 30 minutes), consider it a new session.  
- **Compute Session Duration:**  
  Duration = session_end – session_start.

### Output Dataset  
**Dataset Name:** Session_Dataset  
**Fields:**  
- user_id (normalized)  
- session_id (generated per session)  
- session_start (visitTimestamp)  
- session_end (max(timestamp) in the session)  
- session_duration  

**Dashboard Elements Mapping:**  
- Unique Sessions count  
- Session Duration histogram

---

## 3. Chat Thread & Flow Processing

### Input  
Raw_Cleaned_Log  

### Operations  
- **Grouping by chat_id:**  
  For each chat thread, order events by `chat_seq`.  
- **Flow Construction:**  
  • Identify the sequence of actions per chat: from initial data load (query_type “00_load...”) to user questioning (02_rephrase) and code generation (03_generate_code, which may contain retries) to chart generation (04_generate_run_charts).  
  • Detect multiple consecutive 03_generate_code events as retries before a success or moving to a 04 event.
- **Aggregate Chat Metrics:**  
  • Calculate the total number of queries per chat thread.  
  • Compute the count of retries for code generation within each chat thread.

### Output Dataset  
**Dataset Name:** Chat_Flow_Dataset  
**Fields:**  
- chat_id  
- ordered_events (list/order of query_type events)  
- total_queries (number of events in the chat)  
- retry_count (number of times query_type “03_generate_code” was retried)  

**Dashboard Elements Mapping:**  
- Total Chats Initiated count  
- Funnel Visualization (showing the stepwise flow)  
- Retry Patterns analysis

---

## 4. KPI & Time-Series Aggregation

### Input  
Raw_Cleaned_Log, Session_Dataset, and Chat_Flow_Dataset  

### Operations  
- **Time Bucketing:**  
  Aggregate data by selectable time intervals (daily, weekly, monthly) using the timestamp fields.
- **Metrics Computation:**  
  • Count of LLM calls (all records)  
  • Number of active users per period (distinct user identifiers)  
  • Count of unique sessions (from Session_Dataset)  
  • Count of chats initiated (distinct chat_id)  
  • Distribution of query types (group by query_type)  
  • Feature toggle usage from `enable_chart_generation` and `enable_business_insights` fields  
  • Dataset usage counts from `datasets_names`  

### Output Dataset  
**Dataset Name:** Time_Series_Aggregates  
**Fields:**  
- period (e.g., date, week, or month)  
- total_llm_calls  
- active_users  
- unique_sessions  
- total_chats  
- query_type_counts (map/dictionary or table structure of each query_type and its count)  
- feature_toggle_usage metrics  
- dataset_usage counts  

**Dashboard Elements Mapping:**  
- Time-Series area or line charts (sessions, users, LLM calls)  
- Query Type Distribution visualization  
- Governance panels (usage by domain/dataset)

---

## 5. Error Analysis & Grouping

### Input  
Raw_Cleaned_Log  

### Operations  
- **Identify Errors:**  
  Filter records where `error_message` is not null.
- **Error Grouping:**  
  Group similar error messages into common categories (e.g., code generation failure patterns).  
- **Aggregate Counts:**  
  Count the number of occurrences per error category.

### Output Dataset  
**Dataset Name:** Error_Taxonomy_Dataset  
**Fields:**  
- error_category (grouped/normalized error strings)  
- error_count  

**Dashboard Elements Mapping:**  
- Error & Retry Analysis Panel  
- Display the frequency and taxonomy of errors to pinpoint common issues.

---

## 6. User Message Text Processing for Word-Cloud

### Input  
Raw_Cleaned_Log (user_msg field)

### Operations  
- **Normalization:**  
  Normalize Japanese text (e.g., applying NFKC normalization, converting full-width to half-width characters).
- **Tokenization:**  
  Use a Japanese text-processing library (e.g., Janome or fugashi) to tokenize `user_msg`.  
- **POS Filtering & Stop-Word Removal:**  
  Keep only tokens with relevant parts of speech (e.g., 名詞 (nouns), 動詞 (verbs), 形容詞 (adjectives)).  
  Remove very common Japanese stop words.
- **Frequency Calculation:**  
  Count the occurrences of each token across all user messages.

### Output Dataset  
**Dataset Name:** Word_Cloud_Dataset  
**Fields:**  
- token (the word or term)  
- frequency   

**Dashboard Elements Mapping:**  
- Word-Cloud Panel (display frequently used terms in user messages)

---

## 7. Data Lineage Diagram (Summary)

Below is a simplified overview of how each dataset is derived from the raw log data:

                                        +---------------------+
                                        |  Raw Log Data       |
                                        | (Structured Fields) |
                                        +----------+----------+
                                                   |
                                                   v
                              +--------------------+---------------------+
                              |                                           |
                   [Normalization & Cleaning]                    (Validation)
                              |                                           |
                              v                                           v
                      +---------------+                          +--------------+
                      | Raw_Cleaned_Log  | <--------------+      | (Unified schema) |
                      +---------------+                          +--------------+
                              |
           +------------------+------------------+---------------------+
           |                  |                  |                     |
           v                  v                  v                     v
    [Session Inference]  [Chat Flow]      [KPI Aggregation]        [Error Analysis]
           |                  |                  |                     |
           v                  v                  v                     v
+-----------------+   +-----------------+  +-----------------+  +-----------------+
| Session_Dataset |   | Chat_Flow_Dataset| |Time_Series_Aggregates| |Error_Taxonomy_Dataset|
+-----------------+   +-----------------+  +-----------------+  +-----------------+
                               |
                               +------------+
                                            |
                                            v
                                 [User Message Text Processing]
                                            |
                                            v
                                   +---------------------+
                                   | Word_Cloud_Dataset  |
                                   +---------------------+

---

## 8. Dashboard Elements Mapping Recap

- **Time-Series Charts:**  
  • Source: Time_Series_Aggregates  
  • Elements: Total sessions (from Session_Dataset), active users, LLM calls, and query type distribution.

- **Flow & Funnel Visualizations:**  
  • Source: Chat_Flow_Dataset  
  • Elements: Count of chat threads, retry patterns, and the overall sequence of events.

- **Error & Retry Analysis Panels:**  
  • Source: Error_Taxonomy_Dataset (and aggregated retry data from Chat_Flow_Dataset)  
  • Elements: Error frequencies, retry counts, and error categorization.

- **Governance Panels:**  
  • Source: Time_Series_Aggregates  
  • Elements: Dataset usage and feature toggle breakdown; domain and user type summaries.

- **Word-Cloud:**  
  • Source: Word_Cloud_Dataset  
  • Element: Display of tokenized and frequency-scored words from user_msg (with Japanese language considerations).

---

## 9. Next Steps

1. Develop and test individual transformation scripts for each section (e.g., session inference, chat grouping, KPI aggregation, error grouping, text tokenization).  
2. Schedule integration and run end-to-end tests on a sample of raw data.  
3. Validate that all output datasets match dashboard KPIs expectations and are continuously refreshed (e.g., via nightly jobs).  
4. Integrate the final output datasets with your chosen dashboard tool.

This detailed data lineage and process flow should provide the clarity needed on what to do with the raw data, which operations to perform, the resulting intermediate datasets, and how they connect to each dashboard element.

