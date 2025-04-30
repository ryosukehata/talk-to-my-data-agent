# Product Requirements Document for "Talk to Your Data" Dashboard

## 1. Overview

The “Talk to Your Data” Dashboard is built for the Center of Excellence (CoE) of Digital Transformation. Its purpose is to provide actionable insights into how end users are interacting with a natural language query tool for data analysis. The dashboard aims to:
- Monitor adoption and overall usage of the app.
- Identify and surface friction points (errors, retries, and process delays).
- Ensure proper governance by tracking feature toggles, dataset selection, and user domains.
- Present initial natural language insights (via a Japanese-aware word cloud) to capture the type of questions users are asking.

## 2. Objectives

- Provide clear and interactive views into the app’s performance.
- Deliver detailed drill-downs by time, user group, and dataset usage.
- Track key metrics (sessions, active users, query types, error events, and retries).
- Allow CoE teams to easily identify process bottlenecks and user difficulties.
- Offer a configurable time resolution (daily, weekly, monthly) and filtering capabilities.

## 3. Target Audience

- Digital Transformation CoE members
- Product managers and app administrators
- Data governance and compliance teams

## 4. Dashboard Structure & Layout

The dashboard is envisioned as a multi-panel, responsive web application that accommodates both high-level summary views and detailed analysis pages. Below are the primary pages and their components.

### 4.1. Overall Navigation & Global Elements

- **Header:**  
  • Logo, product name, user profile, logout/link to documentation  
  • Global time selector (date-picker with quick options: Daily, Weekly, Monthly – that updates all panels).

- **Side Navigation Bar:**  
  • Menu items:  
    – Overview Dashboard  
    – User Engagement  
    – Query Flows & Errors  
    – Dataset & Governance  
    – Natural Language Insights  
  • Global Filters:  
    – Domain (drop-down: e.g., a.com, b.com)  
    – User Type (checkboxes: creator, guest)  
    – Data Source (multi-select: database, catalog, etc.)  
    – Query Type (multi-select list)

- **Footer:**  
  • Version information, contact/support details, and last refresh time.

### 4.2. Dashboard Pages & Components

#### Page 1: Overview Dashboard

**Purpose:** Present high-level KPIs and time-series trends so the CoE can quickly assess overall app health.

- **Top KPI Cards (Header Row):**  
  • Unique Sessions  
    – Number computed from session inference (visitTimestamp to last request timestamp per user per session)  
  • Active Users  
    – Distinct count of user identifiers (user_email/userId) per period  
  • Total LLM Calls  
    – Count all log entries (aggregated by query_type events)  
  • Total Chat Threads  
    – Count unique chat_id values  
  • (Optional) Average Session Duration  
    – Computed from first to last event timestamps within a session  
  **UX Details:**  
  – Each card displays the metric value, a small icon, color-coded indicators (green for positive trends, red if below threshold), and a % change compared to the previous period.
  – Cards are clickable. On click, a modal provides historical trends and breakdowns for that KPI.

- **Time Series Trend Chart:**  
  • A multi-line Area or Line Chart showing trends for sessions, active users, and total LLM calls over the selected time period.
  • X-axis: time (based on granularity selector), Y-axis: counts.
  • Hover tooltips display exact figures and % changes.
  • Zoom and pan enabled.

- **Interactive Filter Panel (Side or Top Embedded):**  
  • Adjust the displayed metrics by global filters (Domain, Data Source, Query Type, etc.)
  – Filter changes update all components in real time.

#### Page 2: User Engagement & Query Flow

**Purpose:** Show details around conversation flows and feature usage.

- **Chat Flow Breakdown Panel (Funnel Chart):**  
  • Visualize a typical user journey:
    – Start with “00_load_from_database_callback” → "02_rephrase" → “03_generate_code…” attempts → “04_generate_run_charts_python_code”
  • Highlight average number of retries before successful code generation.
  • Component Details:  
    – Steps are clickable to drill down into the list of logs for that step.
    – A tooltip over each funnel stage shows the conversion rate (% moving to the next stage) and count of events.

- **Query Type Distribution Panel (Bar Chart or Pie Chart):**  
  • Display counts of events by query_type:
    – E.g., “03_generate_code_database (10)”, “03_generate_code_file (7)”, etc.
  • Drilling down:
    – Click a segment to see a time-series breakdown for that specific query_type.

- **Feature Toggle Usage Panel (Stacked Bar or Donut Chart):**  
  • Visual representation of enable_chart_generation and enable_business_insights status.
  • Metrics:
    – Percentage and counts of calls with each toggle on/off (by default, most calls with True).
  • UX: Hover over segments for percentages and numeric details.

- **Session Duration Histogram:**  
  • Histogram showing distribution of session durations (inferred from visitTimestamp to the last request timestamp).
  • Identify outliers (very short or extremely long sessions).
  • UX: Hover tooltips with average duration and quartile stats.

#### Page 3: Quality, Errors & Governance

**Purpose:** Focus on error trends and dataset usage to identify friction points, monitor governance and compliance.

- **Error Analytics Panel (Bar Chart & Table):**  
  • Two parts:  
    – A bar chart summarizing error frequencies grouped by error_message category.
    – A table listing top 5 error messages with occurrence counts and associated chat_id or association_id examples.
  • UX: Clickable error bars to see a list of logs that failed. Filters applied automatically update other charts.

- **Dataset Usage Panel (Horizontal Bar Chart):**  
  • List the most frequently selected datasets based on datasets_names.
  • For each dataset (or combination of datasets), display:
    – Count of times selected.
    – Average query count per dataset.
  • Interaction: Hover to see details and click a bar to drill down to view user and temporal details of queries on that dataset.

- **User Domain & Role Analysis Panel (Stacked Bar Chart):**  
  • Breakdown by domain (e.g., a.com vs. b.com) with detailed splits for userType (creator vs. guest).
  • UX: Filters should allow selection of one or more domains, updating other pages in real time.

#### Page 4: Natural Language & Query Insights

**Purpose:** Provide a glimpse into user intent and queries using a text-based visualization.

- **Japanese-Aware Word Cloud:**
  • Create a dynamic word cloud based on tokenized content from user_msg.
  • Requirements:
    – Use a Japanese morphological analyzer (e.g., Janome, fugashi, or sudachiPy) for tokenization.
    – Filter stop-words and select significant tokens (nouns, adjectives, verbs).
  • UX:
    – Hover reveals the token frequency count.
    – Clicking a token filters the underlying data (e.g., brings up a list of user_msg instances containing that token).
    – Provide manual refresh if needed and allow word size scaling adjustments.

- **Query Example Table:**  
  • A paginated list that shows user_msg samples with associated metadata (user, timestamp, query_type, dataset selected).
  • Supports search and sort functions.
  • UX: “View More” expands into a modal with full conversation history (using chat_id capture).

## 5. Interaction Workflows & UX Details

- **Global Time and Filter Sync:**  
  • The global time selector and filters are persistent across all pages. When a filter is changed, all visual components are refreshed via asynchronous API calls.
- **Drill-Down Behavior:**  
  • Clicking on any chart component (KPI card, bar segment, funnel stage, error bar) opens a drill–down modal or navigates to a detailed report view with a lower level of granularity.
- **Responsive Design:**  
  • The layout should automatically adjust for desktop and tablet screens. Use collapsible side menus on smaller displays.
- **Refresh & Auto-Update:**  
  • The dashboard shows the last update timestamp and supports manual refresh. Ideally, if real-time processing is available, components should auto-refresh every 5–10 minutes.
- **Error Handling & User Feedback:**  
  • When data fails to load (pipeline error or network delay), provide friendly error messages with retry options.  
  • Include a progress indicator during data fetch or re-filtering.

## 6. Mock-Up Layout (Illustrative Example)

```
┌────────────────────────────────────────────────────────────────────────────┐
│                        Header (Logo | Global Time Selector | User)         │
├────────────────────────────────────────────────────────────────────────────┤
│ Sidebar (Filters: Domain, Data Source, Query Type, User Type)   | Overview  │
│                                                               |  Page     │
│                                                               |─────────────────────────────|
│                                                               | KPI Cards (row):          │
│                                                               | - Sessions  - ActiveUsers │
│                                                               | - LLM Calls - Chat Threads│
│                                                               |─────────────────────────────|
│                                                               | Time Series Trend Chart    │
│                                                               |─────────────────────────────|
│                                                               | Query Type Distribution    │
│                                                               | Panel (bar/pie chart)      │
│                                                               |─────────────────────────────|
│                                                               | Funnel Chart / Session     │
│                                                               | Duration Histogram         │
│                                                               |─────────────────────────────|
│                                                               | Word Cloud Panel (NLU)     │
└────────────────────────────────────────────────────────────────────────────┘
```

*Note: The actual UI design should be provided in wireframe/mockup sketches. This diagram simply represents layout regions and the hierarchical structure of components.*

## 7. Acceptance Criteria

- All high-level KPIs and time series trends are accurately computed and reflected.
- Charts and panels update in real time when filters or time granularity selections change.
- Drill-down modals and detailed views correctly reflect the lower-level data behind each metric.
- The word cloud accurately reflects tokenized user_msg text using Japanese morphological analysis.
- The dashboard is responsive and accessible on multiple device types.
- Error handling is in place across all user interactions.