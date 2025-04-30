# Implementation Document: MVP Streamlit App with Trend Granularity Options

## 1. Overview

This document explains how to build an MVP version of the "Talk to Your Data" dashboard using Streamlit. The app will load preprocessed output data (CSV files) from the pipeline and display several key panels – including trend charts that can be toggled by daily, weekly, or monthly aggregates.

We assume the following file structure is in place:

```
output
├── chat_flows.csv
├── daily_active_users.csv
├── daily_chat_metrics.csv
├── daily_dataset_usage.csv
├── daily_error_metrics.csv
├── daily_feature_usage.csv
├── daily_llm_calls.csv
├── daily_query_type_metrics.csv
├── daily_unique_sessions.csv
├── daily_user_governance.csv
├── daily_word_cloud_data.csv
└── sessions.csv
```

In this guide you will learn how to first load these files and then build a Streamlit app that lets users choose their trend granularity (Daily, Weekly, Monthly) and view the corresponding charts.

──────────────────────────────
## 2. Prerequisites and Setup

### 2.1. Environment Setup
- Install Python (version 3.8+ recommended).
- Create and activate a virtual environment:
  ```bash
  python -m venv venv
  source venv/bin/activate    # On Windows: venv\Scripts\activate
  ```

### 2.2. Install Required Python Packages
Install the needed libraries:
```bash
pip install streamlit pandas plotly matplotlib wordcloud
```
These packages are used as follows:
- **Streamlit:** For the web-based dashboard UI.
- **Pandas:** To load and manipulate CSV data.
- **Plotly:** To create interactive trend charts.
- **Matplotlib & WordCloud:** To build and display a word cloud for natural language insights.

──────────────────────────────
## 3. Application Structure

For this MVP, we use a single file (e.g., `main.py`). The file structure:
```
project_root/
 ├── main.py
 ├── requirements.txt
 └── output/
      ├── chat_flows.csv
      ├── daily_active_users.csv
      ├── daily_chat_metrics.csv
      ├── daily_dataset_usage.csv
      ├── daily_error_metrics.csv
      ├── daily_feature_usage.csv
      ├── daily_llm_calls.csv
      ├── daily_query_type_metrics.csv
      ├── daily_unique_sessions.csv
      ├── daily_user_governance.csv
      ├── daily_word_cloud_data.csv
      └── sessions.csv
```
The `requirements.txt` file lists the packages (streamlit, pandas, plotly, matplotlib, wordcloud).

──────────────────────────────
## 4. Detailed Implementation Steps

Below is the complete code for an MVP that supports trend granularity selections. The code is written to be simple enough for a junior developer to follow.

### 4.1. Main Streamlit App (main.py)

Create (or update) `main.py` with the following code:

```python
# main.py

import streamlit as st
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS

# ---------------------------
# Helper function to load CSV data with caching
# ---------------------------
@st.cache_data
def load_csv_data(file_path):
    """Load a CSV file into a pandas DataFrame."""
    return pd.read_csv(file_path)

# ---------------------------
# Load all data files from the output folder
# ---------------------------
def load_all_data():
    data = {}
    data['daily_unique_sessions'] = load_csv_data("output/daily_unique_sessions.csv")
    data['daily_active_users'] = load_csv_data("output/daily_active_users.csv")
    data['daily_llm_calls'] = load_csv_data("output/daily_llm_calls.csv")
    data['daily_query_type_metrics'] = load_csv_data("output/daily_query_type_metrics.csv")
    data['chat_flows'] = load_csv_data("output/chat_flows.csv")
    data['daily_feature_usage'] = load_csv_data("output/daily_feature_usage.csv")
    data['daily_error_metrics'] = load_csv_data("output/daily_error_metrics.csv")
    data['daily_dataset_usage'] = load_csv_data("output/daily_dataset_usage.csv")
    data['daily_user_governance'] = load_csv_data("output/daily_user_governance.csv")
    data['daily_word_cloud_data'] = load_csv_data("output/daily_word_cloud_data.csv")
    return data

data = load_all_data()

# ---------------------------
# Convert 'date' columns to datetime if needed
# ---------------------------
def convert_date_column(df, date_col='date'):
    df[date_col] = pd.to_datetime(df[date_col])
    return df

data['daily_llm_calls'] = convert_date_column(data['daily_llm_calls'])
data['daily_unique_sessions'] = convert_date_column(data['daily_unique_sessions'])
data['daily_active_users'] = convert_date_column(data['daily_active_users'])
# Repeat for other DataFrames that have a date column as necessary

# ---------------------------
# Sidebar: Global Filters and Trend Granularity Selector
# ---------------------------
st.sidebar.header("Global Filters")
# Date filter: now we use it only for context – overall time window might be extended later.
selected_date = st.sidebar.date_input("Select Starting Date", value=pd.to_datetime("today"))
trend_granularity = st.sidebar.selectbox("Select Trend Granularity", ["Daily", "Weekly", "Monthly"])

# ---------------------------
# Utility: Function to aggregate daily data based on granularity
# ---------------------------
def aggregate_trends(df, value_column='value', date_col='date', granularity="Daily"):
    # Ensure date column is datetime
    df = convert_date_column(df, date_col)
    if granularity == "Weekly":
        # Group by week. Reset the index to have a 'date' column.
        agg_df = df.groupby(pd.Grouper(key=date_col, freq='W')).sum().reset_index()
    elif granularity == "Monthly":
        agg_df = df.groupby(pd.Grouper(key=date_col, freq='M')).sum().reset_index()
    else:  # Daily
        agg_df = df
    return agg_df

###########################
## Navigation Tabs
###########################
tabs = st.tabs(["Overview", "User Engagement & Query Flow", "Quality & Governance", "Natural Language Insights"])

##########################################
## Tab 1: Overview Dashboard
##########################################
with tabs[0]:
    st.header("Overview Dashboard")
    
    # KPI Cards in a Row using st.columns
    col1, col2, col3, col4 = st.columns(4)
    
    # For this example, we assume each daily data CSV has a column 'value' with trend metrics.
    unique_sessions_value = data['daily_unique_sessions']['value'].iloc[-1]
    active_users_value = data['daily_active_users']['value'].iloc[-1]
    llm_calls_value = data['daily_llm_calls']['value'].iloc[-1]
    chat_threads_value = data['daily_chat_metrics']['chat_threads'].iloc[-1] if 'chat_threads' in data['daily_chat_metrics'].columns else 0

    with col1:
        st.metric("Unique Sessions", unique_sessions_value)
    with col2:
        st.metric("Active Users", active_users_value)
    with col3:
        st.metric("LLM Calls", llm_calls_value)
    with col4:
        st.metric("Chat Threads", chat_threads_value)

    # Time Series Chart: LLM Calls Trend
    st.subheader("LLM Calls Trend")
    llm_df = aggregate_trends(data['daily_llm_calls'], value_column='value', granularity=trend_granularity)
    fig = px.line(llm_df, x='date', y='value', title=f"LLM Calls Trend ({trend_granularity})")
    st.plotly_chart(fig, use_container_width=True)

##############################################
## Tab 2: User Engagement & Query Flow
##############################################
with tabs[1]:
    st.header("User Engagement & Query Flow")
    
    # Query Type Distribution Bar Chart
    st.subheader("Query Type Distribution")
    fig2 = px.bar(data['daily_query_type_metrics'], x='query_type', y='count', 
                  title="Counts by Query Type")
    st.plotly_chart(fig2, use_container_width=True)

    # Chat Flow & Retry Analysis from chat_flows.csv
    st.subheader("Chat Flow & Retry Analysis")
    st.write("Total Chats:", data['chat_flows']['chat_id'].nunique())
    if 'retry_count' in data['chat_flows'].columns:
       avg_retry = data['chat_flows']['retry_count'].mean()
       st.metric("Average Code Generation Retry", f"{avg_retry:.2f}")
    st.dataframe(data['chat_flows'].head(10))

###########################################
## Tab 3: Quality, Errors & Governance
###########################################
with tabs[2]:
    st.header("Quality, Errors & Governance")

    # Error Metrics Visualization
    st.subheader("Error Metrics")
    fig3 = px.bar(data['daily_error_metrics'], x='error_category', y='count', 
                  title="Error Occurrences by Category")
    st.plotly_chart(fig3, use_container_width=True)

    # Dataset Usage Panel
    st.subheader("Dataset Usage")
    fig4 = px.bar(data['daily_dataset_usage'], x='dataset_name', y='selection_count',
                  title="Dataset Selection Frequency")
    st.plotly_chart(fig4, use_container_width=True)

    # User Domain & Role Breakdown
    st.subheader("User Domain & Role Breakdown")
    fig5 = px.bar(data['daily_user_governance'], x='domain', y='count', color='userType',
                  title="User Domains and Roles")
    st.plotly_chart(fig5, use_container_width=True)

##########################################
## Tab 4: Natural Language Insights
##########################################
with tabs[3]:
    st.header("Natural Language Insights")

    st.subheader("Japanese Word Cloud from User Queries")
    word_freq = {row['token']: row['frequency'] for idx, row in data['daily_word_cloud_data'].iterrows()}
    if word_freq:
        # IMPORTANT: Update 'font_path' with the path to a Japanese TTF font file.
        wordcloud = WordCloud(background_color="white", width=800, height=400,
                              stopwords=STOPWORDS, font_path="path/to/a/japanese/font.ttf").generate_from_frequencies(word_freq)
        fig_wc, ax_wc = plt.subplots(figsize=(10, 4))
        ax_wc.imshow(wordcloud, interpolation='bilinear')
        ax_wc.axis("off")
        st.pyplot(fig_wc)
    else:
        st.write("No word cloud data available.")

    st.subheader("Example User Queries")
    st.dataframe(data['daily_word_cloud_data'].head(10))
```

### 4.2. Code Explanations

1. **Global Trend Granularity Selector:**
   - The sidebar includes a selectbox letting users choose between "Daily", "Weekly", and "Monthly".
   - The selected value is stored in `trend_granularity`.

2. **Aggregation Function:**
   - The `aggregate_trends` function groups the daily data using `pd.Grouper` based on the chosen frequency:
     - `'W'` for weekly.
     - `'M'` for monthly.
     - If "Daily" is selected, no grouping is performed.
   - This function is applied to the daily metrics data (e.g., LLM Calls) before plotting.

3. **Updated Charts:**
   - In the Overview tab, the “LLM Calls Trend” chart calls `aggregate_trends` on the `daily_llm_calls` DataFrame. The x-axis then reflects the aggregated date values based on the selected granularity.
   - Similar techniques can be applied to additional charts as needed.

4. **Other Panels:**
   - The remaining tabs show example aggregations for query type distribution, chat flows, error metrics, dataset usage, and user governance.
   - The Natural Language Insights tab continues to display a Japanese-aware word cloud (ensure you update the font path as needed).

──────────────────────────────
## 5. Running the App

1. Ensure your working directory is the project root (where `main.py` resides).
2. Run the app with:
   ```bash
   streamlit run main.py
   ```
3. Open the browser window and interact with the dashboard. Use the sidebar to select Daily, Weekly, or Monthly trends; the relevant charts will update accordingly.

──────────────────────────────
## 6. Additional Enhancements

- **Data Filtering:**  
  In future iterations, use the global filters (selected date, domains, etc.) to sub-filter each DataFrame.
  
- **Drill-Down Features:**  
  Add clickable elements that open detailed views or popups.
  
- **Advanced Styling:**  
  Consider customizing the app style using Streamlit’s theming or custom CSS.
