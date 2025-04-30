import logging
import random
import uuid
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
from faker import Faker

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

# --- Configuration ---
NUM_USERS = 15
NUM_DAYS = 120  # Approx 3 months (Feb 1st to Apr 30th)
START_DATE = datetime(2025, 1, 1)
AVG_EVENTS_PER_USER_PER_DAY = 5
OUTPUT_FILE = Path(__file__).parent / "input" / "merged_dataset.csv"

# Ensure output directory exists
OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

fake = Faker()

# --- Data Definitions ---
DOMAINS = ["company.com", "research.org", "analytics.corp", "datarobot.com"]
USER_TYPES = ["creator", "guest"]
DATA_SOURCES = ["catalog", "database", "file"]
DATASET_NAMES = [
    "Sales_Data_Q1",
    "Marketing_Campaign_Performance",
    "Customer_Churn_Predict",
    "Inventory_Levels_Daily",
    "Website_Traffic_Logs",
    "HR_Employee_Satisfaction",
    "LENDING_CLUB_PROFILE_JP",
    "LENDING_CLUB_TARGET_JP",
    "LENDING_CLUB_TRANSACTIONS_JP",
    "コーティング製品ブリードアウトmain_train.csv",
    "Predictive AI MLOps Starter Retraining Data 02 [ygu-test] [d4208b4]",
]
QUERY_TYPES_FLOW = [
    "00_registry_download_callback",
    "00_load_from_database_callback",
    "02_rephrase",
    "03_generate_code_file",
    "03_generate_code_database",
    "04_generate_business_analysis",
    "04_generate_run_charts_python_code",
]
SAMPLE_MESSAGES = [
    "show me sales trends",
    "summarize marketing results",
    "predict customer churn",
    "what is the current inventory?",
    "analyze website traffic",
    "employee satisfaction scores?",
    "中身教えて",
    "ブリードアウトが悪化してる根拠あるか？",
    "意外なこと教えて",
    "一番取引金額大きい顧客は誰？",
    "顧客IDかアカウントID、重複なものあるか？",
    "このデータおかしいところあるか？",
]
ERROR_MESSAGES = [
    "InvalidGeneratedCode: Function analyze_data raised an error during execution: 'DataFrame' object has no attribute 'some_method'",
    "InvalidGeneratedCode: Query execution failed: Snowflake error: 100038 (22018): Numerical value 'XYZ' is not recognized",
    "TimeoutError: LLM call timed out",
    "ValidationError: Input data format incorrect",
    "",  # Represents no error
    "",
    "",
    "",
    "",
    "",
]

# --- User Generation ---
users = []
for i in range(NUM_USERS):
    domain = random.choice(DOMAINS)
    user_type = random.choice(USER_TYPES)
    if domain == "datarobot.com" and i < 2:  # Ensure original users are included
        email = "yifu.gu@datarobot.com" if i == 0 else "yifu.gu+demo@datarobot.com"
        user_type = "creator" if i == 0 else "guest"
    else:
        email = f"{fake.user_name()}_{random.randint(1,99)}@{domain}"
        # Ensure yifu users are unique if generated randomly
        while email.startswith("yifu.gu"):
            email = f"{fake.user_name()}_{random.randint(1,99)}@{domain}"

    users.append(
        {
            "user_email": email,
            "userId": str(uuid.uuid4()),  # Consistent User ID
            "userType": user_type,
            "domain": domain,
            "username": email,  # Simple mapping for username
        }
    )

logging.info(f"Generated {len(users)} users.")

# --- Event Generation ---
all_events = []
current_date = START_DATE

for day in range(NUM_DAYS):
    day_str = current_date.strftime("%Y-%m-%d")
    logging.info(f"Generating data for {day_str}...")

    # Select active users for the day
    num_active_users = random.randint(
        1, max(2, int(NUM_USERS * 0.6))
    )  # 60% active users max
    active_users_today = random.sample(users, num_active_users)

    for user in active_users_today:
        num_events_today = max(1, int(random.gauss(AVG_EVENTS_PER_USER_PER_DAY, 2)))
        session_start_time = current_date + timedelta(
            hours=random.randint(8, 18),
            minutes=random.randint(0, 59),
            seconds=random.randint(0, 59),
        )
        visitTimestamp = session_start_time - timedelta(
            minutes=random.randint(1, 10)
        )  # Visit time slightly before session
        current_time = session_start_time
        chat_id = str(uuid.uuid4())
        chat_seq = 0.0
        datasets_this_chat = random.sample(DATASET_NAMES, k=random.randint(1, 3))
        enable_charts = random.choice([True, False])
        enable_insights = random.choice([True, False])
        current_flow = []  # Track sequence of query types in this chat

        for event_num in range(num_events_today):
            current_time += timedelta(
                seconds=random.randint(5, 180)
            )  # Time between events
            association_id = str(uuid.uuid4())
            query_type = random.choice(QUERY_TYPES_FLOW)
            data_source = random.choice(DATA_SOURCES)
            user_msg = ""
            datasets_names_str = str(
                datasets_this_chat
            )  # Keep datasets consistent within chat

            # Make query flow somewhat logical
            if query_type.startswith("02_"):  # Rephrase starts a new flow seq
                chat_seq += 2.0  # Increment by 2 for new user message
                user_msg = random.choice(SAMPLE_MESSAGES)
                current_flow = [query_type]
            elif query_type.startswith("03_") or query_type.startswith("04_"):
                if chat_seq == 0.0:  # Cannot start with code gen or analysis
                    query_type = "02_rephrase"  # Force start with rephrase
                    chat_seq = 2.0
                    user_msg = random.choice(SAMPLE_MESSAGES)
                    current_flow = [query_type]
                else:
                    # Inherit message from the start of the sequence
                    user_msg = next(
                        (
                            e["user_msg"]
                            for e in reversed(all_events)
                            if "chat_id" in e
                            and e["chat_id"] == chat_id
                            and "user_msg" in e
                            and e["user_msg"]
                        ),
                        "Follow up query",
                    )
                    current_flow.append(query_type)
            else:  # Callbacks don't have user messages or belong to chat sequences in the same way
                user_msg = None
                datasets_names_str = None  # Callbacks don't list datasets here
                # Don't add callbacks to current_flow for chat_flows.csv logic

            event = {
                "association_id": association_id,
                "timestamp": current_time.strftime("%Y-%m-%d %H:%M:%S"),
                "user_email": user["user_email"],
                "data_source": data_source,
                "query_type": query_type,
                "user_msg": user_msg,
                "chat_id": (
                    chat_id if query_type.startswith(("02", "03", "04")) else None
                ),
                "chat_seq": (
                    chat_seq if query_type.startswith(("02", "03", "04")) else None
                ),
                "datasets_names": (
                    datasets_names_str
                    if query_type.startswith(("02", "03", "04"))
                    else None
                ),
                "enable_chart_generation": (
                    enable_charts if query_type.startswith(("02", "03", "04")) else None
                ),
                "enable_business_insights": (
                    enable_insights
                    if query_type.startswith(("02", "03", "04"))
                    else None
                ),
                "error_message": (
                    random.choice(ERROR_MESSAGES) if random.random() < 0.1 else None
                ),  # 10% chance of error
                "username": user["username"],
                "visitTimestamp": visitTimestamp.strftime("%Y-%m-%d %H:%M:%S"),
                "userId": user["userId"],
                "userType": user["userType"],
                "domain": user["domain"],
            }
            # Clean up None values before adding
            event_clean = {k: v for k, v in event.items() if v is not None}
            all_events.append(event_clean)

    current_date += timedelta(days=1)

# --- Create DataFrame and Save ---
df = pd.DataFrame(all_events)

# Ensure correct column order as expected by the pipeline (based on original file)
expected_cols = [
    "association_id",
    "timestamp",
    "user_email",
    "data_source",
    "query_type",
    "user_msg",
    "chat_id",
    "chat_seq",
    "datasets_names",
    "enable_chart_generation",
    "enable_business_insights",
    "error_message",
    "username",
    "visitTimestamp",
    "userId",
    "userType",
    "domain",
]

# Add missing columns with default NaN or appropriate type if needed
for col in expected_cols:
    if col not in df.columns:
        # Handle specific defaults if necessary, e.g., boolean columns
        if col in ["enable_chart_generation", "enable_business_insights"]:
            df[col] = pd.NA  # Use pandas NA for missing booleans if appropriate
        else:
            df[col] = pd.NA  # Default to pandas NA

df = df[expected_cols]  # Reorder and select only expected columns

# Convert timestamp columns to datetime objects before saving (optional but good practice)
df["timestamp"] = pd.to_datetime(df["timestamp"])
df["visitTimestamp"] = pd.to_datetime(df["visitTimestamp"])

# Sort by timestamp
df = df.sort_values(by="timestamp").reset_index(drop=True)

logging.info(f"Generated {len(df)} events.")
logging.info(f"Saving generated data to {OUTPUT_FILE}...")

# Save to CSV
df.to_csv(OUTPUT_FILE, index=False, date_format="%Y-%m-%d %H:%M:%S")

logging.info("--- Dummy Data Generation Complete ---")
