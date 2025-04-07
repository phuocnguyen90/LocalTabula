# app.py
import streamlit as st
import pandas as pd
import os

# Import functions from utils.py
from utils import (
    setup_environment,
    get_db_connection,
    init_qdrant_client,
    get_llm_wrapper,
    get_cached_aux_models,
    get_schema_info,
    process_uploaded_data,
    process_natural_language_query,
    read_google_sheet,
    table_exists,
)
import logging
# --- Custom Logging Filter ---
class TorchClassesPathFilter(logging.Filter):
    """Filter out annoying 'Examining the path of torch.classes' errors."""
    def filter(self, record):
        # Check if the message originates from the specific problematic check
        # and contains the known error text.
        is_watcher_log = record.name.startswith("streamlit.watcher") # Check logger name
        msg = record.getMessage()
        is_problem_msg = "Examining the path of torch.classes raised" in msg or \
                         "Tried to instantiate class '__path__._path'" in msg or \
                         "no running event loop" in msg # Include the asyncio one too

        # Suppress (return False) only if it's from the watcher AND contains the specific error texts
        # Allow other messages from the watcher and all messages from other loggers
        # Careful: Ensure the logger name check is accurate for your setup if errors persist
        # Sometimes these errors might bubble up to the root logger if not handled.
        # If suppressing based on name fails, rely only on message content check.
        # return not is_problem_msg # Alternative: Suppress regardless of source if message matches
        return not (is_watcher_log and is_problem_msg)

# --- Configure Logging ---
# Get the root logger or a specific Streamlit logger if identifiable
# Adding to root logger is often effective for these kinds of dispersed warnings
logger = logging.getLogger() # Get root logger
# logger = logging.getLogger('streamlit') # Or try the specific streamlit logger

# Set basic logging level (adjust as needed for your own debugging)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - [%(name)s] %(message)s')

# Add the custom filter
# Check if filter already added to prevent duplicates on Streamlit reruns
filter_name = "torch_classes_filter"
if not any(f.name == filter_name for f in logger.filters):
    custom_filter = TorchClassesPathFilter(name=filter_name)
    logger.addFilter(custom_filter)
    logging.info(f"Added '{filter_name}' to logger '{logger.name}' to suppress specific watcher errors.")
else:
    logging.debug(f"Filter '{filter_name}' already exists on logger '{logger.name}'.")
# --- End Logging Setup ---


# --- Call set_page_config FIRST ---
st.set_page_config(layout="wide", page_title="Chat with Your Tabular Data")

# --- Setup Environment and Database Path ---
# This function now handles path logic and directory creation
DB_PATH = setup_environment()

# Halt if environment setup failed (e.g., couldn't create directory)
if not DB_PATH:
    st.error("CRITICAL ERROR: Failed to initialize application environment (check logs/permissions). Application cannot start.")
    st.stop() # Stop script execution


# Order matters less now as they are independent cached resources
db_conn = get_db_connection(DB_PATH)
qdrant_client = init_qdrant_client()
llm_wrapper = get_llm_wrapper()         # Get the LLM wrapper instance
aux_models_dict = get_cached_aux_models() # Get the aux models dictionary

# --- Perform Readiness Checks ---
db_ready = db_conn is not None
qdrant_ready = qdrant_client is not None
llm_ready = llm_wrapper and llm_wrapper.is_ready
aux_models_ready = aux_models_dict and aux_models_dict.get("status") == "loaded"
core_resources_ok = db_ready and qdrant_ready and llm_ready and aux_models_ready
# --- Initialize Session State ---
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Hi! Load data via sidebar, then ask questions."}]
if "schema" not in st.session_state:
    # Get initial schema only if DB connection was successful
    st.session_state.schema = get_schema_info(db_conn) if db_conn else {}
if "confirm_replace_needed" not in st.session_state: st.session_state.confirm_replace_needed = False
if "confirm_replace_details" not in st.session_state: st.session_state.confirm_replace_details = {}
if "process_now_confirmed" not in st.session_state: st.session_state.process_now_confirmed = False # Ensure reset
# --- Initialize Backend Resources ---


# --- Sidebar for Data Input and Management ---

with st.sidebar:
    st.title("📄 Data Input & Management")
    st.subheader("Status")
    # Status checks based on readiness flags
    if db_ready: st.success(f"DB Connected: `{os.path.basename(DB_PATH)}`")
    else: st.error(f"DB Connection Failed.")
    if qdrant_ready: st.success("Qdrant Client Initialized.")
    else: st.error("Qdrant Client Failed.")
    if llm_ready: st.success(f"LLM Ready (Mode: {llm_wrapper.mode})")
    else: st.error(f"Core LLM Failed to Initialize.")
    if aux_models_ready: st.success("Aux Models (SQL/Embed) Loaded.")
    else: st.error(f"Aux Models Failed to Load.") # Error logged in utils

    st.markdown("---")
    # --- Load Data Section ---
    st.header("1. Load Data Source")
    uploaded_file = st.file_uploader("Upload Excel File (.xlsx, .xls)", type=["xlsx", "xls"])
    st.markdown("<p style='text-align: center; color: grey;'>OR</p>", unsafe_allow_html=True)
    gsheet_published_url = st.text_input("Paste Published Google Sheet CSV URL", help="Use File > Share > Publish to web > CSV format")

    st.markdown("---")
    st.header("2. Specify Table Name")
    table_name_input = st.text_input("Enter Table Name for Database", key="table_name_input_key")

    st.markdown("---")
    st.header("3. Process Data")

    # --- Confirmation Logic ---
    confirm_needed = st.session_state.get("confirm_replace_needed", False)
    confirm_details = st.session_state.get("confirm_replace_details", {})

    if confirm_needed and confirm_details.get("table_name") == table_name_input:
        st.warning(f"Table `{table_name_input}` already exists.")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Yes, Replace Table", key="confirm_yes"):
                st.session_state.confirm_replace_needed = False # Reset flag
                # Trigger processing with confirmation=True
                # Store confirmation state temporarily to use after button click
                st.session_state.process_now_confirmed = True
                st.rerun() # Rerun to proceed with processing immediately

        with col2:
            if st.button("No, Cancel", key="confirm_no"):
                st.session_state.confirm_replace_needed = False
                st.session_state.confirm_replace_details = {}
                st.info("Operation cancelled.")
                st.rerun() # Rerun to clear buttons

    else: # Show the main process button if no confirmation is pending
        # ... (Check essential resources and disable button if needed) ...
        
        process_button_disabled = not core_resources_ok # or not table_name_input etc.
        disabled_reason = "Processing disabled: Core resources failed." if not core_resources_ok else ""
        if not table_name_input:
             process_button_disabled = True
             disabled_reason += " Enter a table name."
        if not uploaded_file and not gsheet_published_url:
             process_button_disabled = True
             disabled_reason += " Select a data source."

        if disabled_reason: st.warning(disabled_reason.strip())

        process_clicked = st.button("Load and Process Data", disabled=process_button_disabled, key="process_data_button")

        # Check if we just confirmed and should proceed
        if st.session_state.get("process_now_confirmed", False):
            process_clicked = True # Treat as if button was just clicked
            st.session_state.process_now_confirmed = False # Reset flag

        if process_clicked:
            table_name_to_process = table_name_input # Use current input value
            
            replace_is_confirmed = not st.session_state.get("confirm_replace_needed", False) # If confirm needed was false, it's confirmed implicitly or first time

            # Check if table exists *now* before calling process_uploaded_data
            if db_conn and table_exists(conn=db_conn, table_name=table_name_to_process) and not replace_is_confirmed:
                # Table exists, but confirmation flow wasn't completed or was reset. Need confirmation.
                st.session_state.confirm_replace_needed = True
                st.session_state.confirm_replace_details = {"table_name": table_name_to_process}
                st.rerun()
            else:
                # Proceed with processing (either table doesn't exist or confirmation was given/not needed)
                with st.spinner("Processing data..."):
                    success, message = process_uploaded_data(
                        uploaded_file, gsheet_published_url, table_name_to_process,
                        db_conn, llm_wrapper, aux_models_dict, qdrant_client,
                        replace_confirmed=True 
                    )
                if message: st.info(f"Processing Log:\n```\n{message}\n```")
                if success:
                    st.session_state.schema = get_schema_info(db_conn) # Refresh schema
                    st.success(f"Data processing completed for table `{table_name_to_process}`.")
                    # Clear confirmation state if processing was successful
                    st.session_state.confirm_replace_needed = False
                    st.session_state.confirm_replace_details = {}
                    st.rerun()
                else:
                    st.error(f"Data processing failed for table `{table_name_to_process}`. See logs.")
                    # Optionally keep confirmation state? Or clear? Let's clear.
                    st.session_state.confirm_replace_needed = False
                    st.session_state.confirm_replace_details = {}
                    # Don't rerun on failure, let user see error

    st.markdown("---")
    st.header("ℹ️ Current Database Schema")
    # (Schema display logic remains the same)
    if not db_conn: st.warning("Database not connected.")
    elif not st.session_state.schema: st.info("No tables found. Load data first.")
    else:
        for table, columns in st.session_state.schema.items():
            with st.expander(f"Table: `{table}`"):
                st.write("Columns:", ", ".join([f"`{col}`" for col in columns]))


# --- Main Chat Area ---
st.title("💬 Chat with Your Data")
st.markdown("Ask questions in natural language about the loaded data.")
# --- DEBUGGING LINE ---
st.sidebar.subheader("DEBUG: Session Messages") # Display in sidebar to avoid clutter
st.sidebar.json(st.session_state.messages) # Show the JSON representation
# Or print to console:
# print("DEBUG: Current messages:", st.session_state.messages)
# --- END DEBUGGING LINE ---


# Display chat messages from history
for message in st.session_state.messages:
    # Add a check to handle non-dictionary items gracefully
    if isinstance(message, dict) and "role" in message:
        with st.chat_message(message["role"]):
            content = message.get("content")
            # Display based on content type (DataFrame, List, or String)
            if isinstance(content, pd.DataFrame):
                st.dataframe(content, use_container_width=True)
            elif isinstance(content, list):
                for item in content:
                    st.markdown(f"- {item}")
            else:
                st.markdown(str(content)) # Ensure content is string
    else:
        # If it's not a dict or lacks 'role', display a warning/debug message
        st.warning(f"Skipping invalid message format in history: `{type(message)}`, Value: `{message}`")
        print(f"WARNING: Invalid message format found: {type(message)}, Value: {message}") # Also print to console
# Accept user input (disable if essential resources missing)
chat_input_disabled = not core_resources_ok
prompt_disabled_reason = "Chat disabled: DB, Qdrant, or Models failed." if chat_input_disabled else "Your question here..."

if prompt := st.chat_input(prompt_disabled_reason, disabled=chat_input_disabled):
    # (Chat processing logic remains the same)
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        message_placeholder.markdown("Thinking...")
        # Check resources again just before processing
        if not core_resources_ok:
             message_placeholder.error("Cannot process query: Essential resources (DB, Qdrant, Models) are not available.")
             st.session_state.messages.append({"role": "assistant", "content": "Error: Cannot process query due to initialization failure."})
        else:
            with st.status("Processing your query...", expanded=False) as status:
                result = process_natural_language_query(
                    prompt, db_conn, st.session_state.schema, llm_wrapper, aux_models_dict, qdrant_client
                )
                # ... (Result handling logic is the same) ...
                if result["status"] == "success":
                    status.update(label="Query processed!", state="complete", expanded=False)
                    # ... (display SQL/Semantic results as before) ...
                    response_content = result.get("data")
                    summary = result.get("message", "Processing complete.")
                    history_content = summary
                    if result["type"] == "SQL":
                        if isinstance(response_content, pd.DataFrame):
                            history_content = f"Found {len(response_content)} results via SQL."
                            # Display DataFrame WITHIN the chat message
                            st.dataframe(response_content, use_container_width=True)
                            message_placeholder.empty() # Clear "Thinking..."
                            # --- MOVE EXPANDER OUTSIDE or DISPLAY DIFFERENTLY ---
                            # Option A: Display SQL outside the chat bubble (simpler)
                            # (Code below would go AFTER the `with st.chat_message(...)` block if desired)

                            # Option B: Display inside but check context carefully
                            # If the parent is st.status, it might be okay? Test it.
                            # If the parent is another expander, DEFINITELY move it.

                        else: # Handle unexpected SQL data format
                            history_content = "Received unexpected data format for SQL result."
                            message_placeholder.warning(history_content)
                    elif result["type"] == "SEMANTIC":
                         # ... (Semantic display) ...
                         if isinstance(response_content, list):
                             history_content = f"Found {len(response_content)} semantic results. Details above."
                             for item in response_content: st.markdown(f"- {item}")
                             message_placeholder.empty()
                         else:
                             history_content = "Received unexpected data format for Semantic result."
                             message_placeholder.warning(history_content)
                    else:
                        history_content = str(response_content)
                        message_placeholder.markdown(history_content)
                    st.session_state.messages.append({"role": "assistant", "content": history_content})

                else: # Handle errors
                    status.update(label="Query failed", state="error", expanded=True)
                    error_message = result.get("message", "An unknown error occurred.")
                    message_placeholder.error(error_message)
                    st.session_state.messages.append({"role": "assistant", "content": f"Error: {error_message}"})

# --- Footer ---
st.markdown("---")
st.caption("Powered by Streamlit, SQLite, Qdrant, Google Auth, etc. (Simulated Models)")