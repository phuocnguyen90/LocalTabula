# app.py
import streamlit as st
import pandas as pd
import os
import time
import logging
import torch
torch.classes.__path__ = [] # Clear the path to avoid loading Torch classes

# Import functions from utils.py
from utils.utils import (
    setup_environment, get_db_connection, init_qdrant_client, get_llm_wrapper,
    get_cached_aux_models, get_schema_info, process_uploaded_data,
    process_natural_language_query, table_exists,
    get_sqlite_table_row_count, get_qdrant_collection_info,
    reindex_table, delete_table_data,
    derive_requirements_from_history,
    QDRANT_COLLECTION_PREFIX
)

# --- Config Logging & Filter ---

# --- Setup Environment and Database Path ---
DB_PATH = setup_environment()
if not DB_PATH:
    st.error("CRITICAL ERROR: Failed environment setup.")
    st.stop()

# --- Initialize Backend Resources ---
db_conn = get_db_connection(DB_PATH)
qdrant_client = init_qdrant_client()
llm_wrapper = get_llm_wrapper()
aux_models = get_cached_aux_models()

# --- Check if GPU is available ---
if torch.cuda.is_available():
    logging.info("GPU is available.")
else:
    logging.info("GPU is not available. Using CPU.")


# --- Perform Readiness Checks ---

aux_models_ready = isinstance(aux_models, dict) and aux_models.get("status") == "loaded"
db_ready = db_conn is not None
qdrant_ready = qdrant_client is not None
llm_ready = llm_wrapper and llm_wrapper.is_ready
aux_models_ready = aux_models and aux_models.get("status") == "loaded"

# Processing requires DB, Qdrant, Embedder(Aux), optionally LLM for column suggestions
processing_resources_ok = db_ready and qdrant_ready and aux_models_ready  # Add llm_ready if needed
# Chat requires DB, Qdrant, LLM, Aux (SQL/Embedder)
core_resources_ok = db_ready and qdrant_ready and llm_ready and aux_models_ready

# Provide User Feedback on Readiness
if not core_resources_ok:
    st.warning("Core resources (DB, Qdrant, LLM, Models) are not fully loaded. Chat functionality may be limited or unavailable.", icon="⚠️")
    # Be more specific if possible
    if not db_ready: st.error("Database connection failed.")
    if not qdrant_ready: st.error("Qdrant vector store failed to initialize.")
    if not llm_ready: st.error("LLM wrapper failed to initialize or model not loaded.")
    if not aux_models_ready:
        err = aux_models.get('error_message', 'Check logs') if isinstance(aux_models, dict) else "Models dict not loaded"
        st.error(f"Auxiliary models (Embedder/SQL) failed to load: {err}")

# --- Initialize Session State ---
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Hi! Load data via the Data Management tab, then ask questions here."}]
if "schema" not in st.session_state:
    raw_schema = get_schema_info(db_conn) if db_ready else {}
    # Ensure db_conn is valid before getting schema
    st.session_state.schema = {"default": raw_schema}
    logging.info(f"Initial schema loaded: {list(st.session_state.schema.keys())}")
# Flags for data loading confirmation workflow
if "confirm_replace_needed" not in st.session_state:
    st.session_state.confirm_replace_needed = False
if "confirm_replace_details" not in st.session_state:
    st.session_state.confirm_replace_details = {} # Stores {'table_name': name}
if "process_now_confirmed" not in st.session_state:
    st.session_state.process_now_confirmed = False # Signals processing can proceed

# State for management tab actions
if "selected_table_manage" not in st.session_state:
     # Initialize safely, considering schema might be empty initially
    st.session_state.selected_table_manage = next(iter(st.session_state.schema), None)
if "confirm_delete_table" not in st.session_state:
    st.session_state.confirm_delete_table = None  # Stores table name pending delete confirmation

# --- Define Tabs ---
tab_chat, tab_data_manage = st.tabs(["💬 Chat", "💾 Data Management"])

# --- Chat Tab ---
with tab_chat:
    st.header("Chat with Your Data")
    st.markdown("Ask questions in natural language about the loaded data.")

    # --- Display Chat History ---
    current_messages = st.session_state.get("messages", [])
    for i, message in enumerate(current_messages): # Use enumerate if needing index
        with st.chat_message(message["role"]):
            content = message.get("content") # Main summary/response/error
            raw_display_data = message.get("raw_display_data") # Dict with 'sql' and 'semantic'
            generated_sql = message.get("generated_sql")

            # Display the main message
            st.markdown(str(content))

            # --- Expanders for Raw Data and SQL ---
            expander_key_base = f"detail_expander_{i}" # Unique key prefix per message

            # Show Raw SQL Data if available
            sql_data = raw_display_data.get("sql") if isinstance(raw_display_data, dict) else None
            if sql_data is not None and not sql_data.empty:
                with st.expander("Show Raw SQL Data", expanded=False):
                    st.dataframe(sql_data, use_container_width=True)
            elif sql_data is not None and sql_data.empty:
                with st.expander("Raw SQL Data", expanded=False):                
                    st.caption("SQL query returned no rows.")


            # Show Raw Semantic Data if available
            semantic_data = raw_display_data.get("semantic") if isinstance(raw_display_data, dict) else None
            if semantic_data and isinstance(semantic_data, list):
                # Filter out potential "no matches" messages before showing expander
                valid_semantic_hits = [hit for hit in semantic_data if "No relevant matches found" not in hit]
                if valid_semantic_hits:
                    with st.expander("Show Semantic Search Snippets", expanded=False):
                        for item in valid_semantic_hits:
                                st.markdown(f"```\n{item}\n```") # Show in code block for clarity
                elif "No relevant matches found" in semantic_data[0]: # Check if the only item is the "no matches" message
                    with st.expander("Semantic Search Snippets", expanded=False):
                        st.caption("Semantic search returned no relevant matches.")


            # Show Generated SQL if available
            if generated_sql and "-- Error" not in generated_sql: # Don't show if generation failed
                 with st.expander("Show Executed SQL Query", expanded=False):
                      st.code(generated_sql, language="sql")

    # --- Chat Input ---
    if prompt := st.chat_input("Your question here...", disabled=not core_resources_ok, key="chat_prompt"):
        if not core_resources_ok:
            st.error("Cannot process query: Core resources are not ready.")
        else:
            # Append user message
            st.session_state.messages.append({"role": "user", "content": prompt})
            st.rerun()  # Rerun to display user message immediately

    # Process the latest user message if it exists
    if st.session_state.messages and st.session_state.messages[-1]["role"] == "user":
        user_prompt = st.session_state.messages[-1]["content"]
        # Assume st.session_state.messages contains the conversation history
        refined_requirements = derive_requirements_from_history(st.session_state.messages, llm_wrapper)

        # Optionally, store the refined requirements for debugging:
        st.session_state.refined_requirements = refined_requirements
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            message_placeholder.markdown("Thinking...")
            with st.status("Processing query...", expanded=False) as status:
                # --- Add Type Check Before Call ---
                print(f"DEBUG [app.py Process Data]: Type of aux_models before passing: {type(aux_models)}")
                if not isinstance(aux_models, dict): print("ERROR: aux_models IS STRING before process_uploaded_data!")
                result = process_natural_language_query(
                    original_query=refined_requirements,
                    conn=db_conn,
                    full_schema=st.session_state.schema,
                    llm_wrapper=llm_wrapper,
                    aux_models=aux_models,
                    qdrant_client=qdrant_client,
                    max_retries=1
                )
                message_for_history = {
                  "role": "assistant",
                  "content": result.get("message", "Error: No response message generated."),
                  "raw_display_data": result.get("raw_display_data"), # Store dict
                  "generated_sql": result.get("generated_sql"),
                  # Optionally store status/type for debugging
                  "status": result.get("status"),
                  "type": result.get("type")
                }
                st.session_state.messages[-1] = message_for_history # Replace user msg


                # Display the main message (summary) placeholder update is enough before rerun
                if result.get("status") == "success":
                    status.update(label="Query processed!", state="complete")
                    message_placeholder.markdown(message_for_history["content"])
                else:
                    status.update(label="Query failed", state="error", expanded=True)
                    message_placeholder.error(message_for_history["content"])
                st.rerun() # Rerun to display fully with expanders

# --- Data Management Tab ---
with tab_data_manage:
    st.header("Manage Data Sources")

    # --- Section 1: Load New Data ---
    with st.expander("Load New Data", expanded=True):
        uploaded_file = st.file_uploader("Upload Excel File (.xlsx, .xls)", type=["xlsx", "xls"], key="data_upload")
        st.markdown("<p style='text-align: center; color: grey;'>OR</p>", unsafe_allow_html=True)
        gsheet_published_url = st.text_input("Paste Published Google Sheet CSV URL", placeholder="https://docs.google.com/spreadsheets/d/.../pub?output=csv", help="Use File > Share > Publish to web > CSV format", key="data_gsheet")
        table_name_input = st.text_input("Enter Table Name for Database", help="Alphanumeric chars and underscores. Replaces existing table if name conflicts.", key="data_table_name")

        # Get current state for confirmation logic
        confirm_needed = st.session_state.get("confirm_replace_needed", False)
        confirm_details = st.session_state.get("confirm_replace_details", {})
        entered_table_name = table_name_input.strip() if table_name_input else ""

        # Input validation for table name
        is_valid_table_name = entered_table_name.isidentifier() if entered_table_name else False
        if entered_table_name and not is_valid_table_name:
            st.warning("Invalid table name. Use letters, numbers, and underscores. Cannot start with a number.", icon="⚠️")

        # Check if confirmation is needed for *this specific table name*
        show_confirmation_buttons = confirm_needed and confirm_details.get("table_name") == entered_table_name and entered_table_name

        if show_confirmation_buttons:
            st.warning(f"Table `{entered_table_name}` already exists. Replacing it will delete and reload the data and vector index.", icon="⚠️")
            col1_confirm, col2_confirm = st.columns(2)
            with col1_confirm:
                if st.button("Yes, Replace Table", key="confirm_replace_yes", type="primary"):
                    st.session_state.confirm_replace_needed = False
                    st.session_state.process_now_confirmed = True # Signal to process
                    # No need to set confirm_replace_details here, process flow will use input
                    logging.info(f"User confirmed replacement for table: {entered_table_name}")
                    st.rerun()
            with col2_confirm:
                if st.button("No, Cancel Load", key="confirm_replace_no"):
                    st.session_state.confirm_replace_needed = False
                    st.session_state.confirm_replace_details = {}
                    st.session_state.process_now_confirmed = False
                    st.info("Load operation cancelled.")
                    logging.info(f"User cancelled replacement for table: {entered_table_name}")
                    # Clear inputs? Optional, might be annoying for user
                    # st.session_state.data_upload = None
                    # st.session_state.data_gsheet = ""
                    # st.session_state.data_table_name = ""
                    st.rerun()
        else:
            # Show primary load button only if confirmation isn't actively being shown
            load_disabled = (
                not processing_resources_ok
                or not is_valid_table_name # Check validity
                or (not uploaded_file and not gsheet_published_url)
                or confirm_needed # Disable if confirmation is pending for a *different* table
            )
            if st.button("Load and Process Data", disabled=load_disabled, key="process_data_button"):
                st.session_state.process_now_confirmed = True # Signal intent to process
                logging.info(f"Load button clicked for table: {entered_table_name}")
                st.rerun() # Rerun to trigger the processing logic below

        # --- Processing Logic ---
        # This block runs *after* the button click + rerun, or after confirmation + rerun
        if st.session_state.get("process_now_confirmed", False):
            # Check if confirmation is needed *now* (table exists and wasn't just confirmed)
            table_exists_check = db_ready and is_valid_table_name and table_exists(conn=db_conn, table_name=entered_table_name)

            # Trigger confirmation if: table exists AND confirmation isn't already active for THIS table
            if table_exists_check and not show_confirmation_buttons:
                logging.info(f"Table '{entered_table_name}' exists, triggering confirmation request.")
                st.session_state.confirm_replace_needed = True
                st.session_state.confirm_replace_details = {"table_name": entered_table_name}
                st.session_state.process_now_confirmed = False # Wait for user confirmation
                st.rerun()
            # Proceed with processing if: resources OK AND (table doesn't exist OR confirmation was handled)
            elif processing_resources_ok and is_valid_table_name and (not table_exists_check or show_confirmation_buttons):
                 # Consume the flag now that we are processing
                st.session_state.process_now_confirmed = False
                logging.info(f"Proceeding with data processing for table: {entered_table_name}")
                with st.spinner(f"Processing data for `{entered_table_name}`... This might take time."):
                    # Ensure aux_models is the expected dict
                    if not isinstance(aux_models, dict):
                         st.error(f"CRITICAL: Aux models format error before processing. Type: {type(aux_models)}")
                         logging.error(f"Aux models format error before processing. Type: {type(aux_models)}")
                         # Set success to False or handle appropriately
                         success = False
                         message = "Internal configuration error: Models not loaded correctly."
                    else:
                        # Pass llm_wrapper for column suggestion during processing
                        success, message = process_uploaded_data(
                            uploaded_file, gsheet_published_url, entered_table_name,
                            db_conn, llm_wrapper, aux_models, qdrant_client,
                            replace_confirmed=True # Let utils handle the replacement logic internally
                        )

                        # --- MODIFIED Log Display ---
                        if message:
                            st.subheader("Processing Log") # Add a heading for clarity
                            st.text(message) # Display log directly using st.text
                        # --- End MODIFICATION ---

                        if success:
                            st.success(f"Data processing completed successfully for `{entered_table_name}`.")
                            st.session_state.schema = get_schema_info(db_conn) # Refresh schema
                            logging.info(f"Schema refreshed after processing. Tables: {list(st.session_state.schema.keys())}")
                            st.session_state.confirm_replace_needed = False
                            st.session_state.confirm_replace_details = {}
                            st.rerun() # Rerun to update overview and inputs
                        else:
                            st.error(f"Data processing failed for `{entered_table_name}`. See log above for details.")
                            st.session_state.confirm_replace_needed = False
                            st.session_state.confirm_replace_details = {}
                            # Do not clear inputs on failure, user might want to retry
            elif not is_valid_table_name and entered_table_name:
                 # This case should be caught by disabled button, but added safety
                 st.warning("Please enter a valid table name.")
                 st.session_state.process_now_confirmed = False # Reset flag
            elif not processing_resources_ok:
                 st.error("Cannot process data: Core resources not ready.")
                 st.session_state.process_now_confirmed = False # Reset flag


    st.markdown("---")

    # --- Section 2: Database Overview ---
    st.subheader("Database Status & Overview")
    # Refresh schema from DB connection if ready
    if db_ready:
        st.session_state.schema = get_schema_info(db_conn)
    current_schema = st.session_state.schema # Use the potentially updated schema

    if not db_ready:
        st.warning("Database not connected.")
    elif not current_schema:
        st.info("No tables found in the database. Load data using the section above.")
    else:
        st.write(f"**Tables found:** {len(current_schema)}")
        overview_data = []
        for table_name, columns in current_schema.items():
            row_count = get_sqlite_table_row_count(db_conn, table_name)
            collection_name = f"{QDRANT_COLLECTION_PREFIX}{table_name}"
            # Use the new helper to fetch the most recent collection info for this table.
            qdrant_info = get_qdrant_collection_info(qdrant_client, collection_name ) if qdrant_ready else None
            
            # Display the collection name if found, else note as "Not Found"
            collection_display = collection_name if qdrant_info else "Not Found"
            points = qdrant_info.get("points_count") if qdrant_info else None
            dim    = qdrant_info.get("vector_size")   if qdrant_info else None
            overview_data.append({
                "Table Name":        table_name,
                "Columns":           len(columns),
                "SQLite Rows":       row_count if row_count is not None else "",
                "Vector Collection": collection_display,
                "Vector Points":     points,
                "Vector Dim":        dim
            })
        df = pd.DataFrame(overview_data)
        df["Vector Points"] = df["Vector Points"].astype("Int64")
        df["Vector Dim"]    = df["Vector Dim"].astype("Int64")
        st.dataframe(df, use_container_width=True, hide_index=True)\
        


        # Detailed view per table
        with st.expander("Show Table Schema Details"):
            for table_name, columns in current_schema.items():
                st.markdown(f"**Table: `{table_name}`**")
                cols_md = "\n".join([f"- `{col}`" for col in columns])
                st.markdown(cols_md)
    st.markdown("---")

    # --- Section 3: Manage Existing Tables ---
    st.subheader("Manage Existing Tables")
    if not current_schema:
        st.info("Load data first to manage tables.")
    else:
        table_options = sorted(list(current_schema.keys())) # Sort for consistency

        # --- Logic to determine default index for selectbox ---
        default_index = 0 # Default to first option if no state or state invalid
        selected_table_from_state = st.session_state.get("selected_table_manage")

        if selected_table_from_state in table_options:
            try:
                default_index = table_options.index(selected_table_from_state)
            except ValueError:
                logging.warning(f"Value '{selected_table_from_state}' in session state but not in current options. Defaulting.")
                default_index = 0
                # Optionally clear the invalid state value?
                # st.session_state.selected_table_manage = table_options[0] if table_options else None
        elif table_options:
            # If state value is not valid (e.g., None or deleted table), set state to first valid option
            st.session_state.selected_table_manage = table_options[0]
            default_index = 0
        else: # No tables left
             st.session_state.selected_table_manage = None # Ensure state is None if no options

        # --- Instantiate the selectbox ---
        if not table_options:
             st.caption("No tables available to manage.")
        else:
            selected_table = st.selectbox(
                "Select Table to Manage:",
                options=table_options,
                index=default_index, # Use calculated default index
                key="selected_table_manage" # Key links to session state
            )

            # Ensure the state is updated if the user changes the selection
            # This happens automatically due to the key, but good to be aware

            if selected_table: # Should always be true if table_options is not empty
                st.markdown(f"**Actions for table: `{selected_table}`**")
                col1_manage, col2_manage, col3_manage = st.columns(3)

                with col1_manage: # View Data
                    if st.button(f"View Sample Data", key=f"view_{selected_table}"):
                        if db_ready:
                            try:
                                # Fetch sample data safely, quoting table name
                                sample_df = pd.read_sql_query(f'SELECT * FROM "{selected_table}" LIMIT 10', db_conn)
                                if sample_df.empty:
                                     st.caption(f"Table `{selected_table}` is empty.")
                                else:
                                     st.dataframe(sample_df, use_container_width=True, hide_index=True)
                            except Exception as e:
                                st.error(f"Failed to read sample data for '{selected_table}': {e}")
                                logging.error(f"Error reading sample data for {selected_table}: {e}", exc_info=True)
                        else:
                             st.error("Database not connected.")


                with col2_manage: # Re-index
                    reindex_disabled = not (processing_resources_ok and llm_ready)
                    if st.button(f"🔄 Re-index Vectors", key=f"reindex_{selected_table}", help="Re-calculates and replaces vector embeddings for this table using current models.", disabled=reindex_disabled):
                        with st.spinner(f"Re-indexing '{selected_table}'... This may take time."):
                            # Ensure aux_models format before calling
                            if not isinstance(aux_models, dict):
                                st.error(f"CRITICAL: Aux models format error before re-indexing. Type: {type(aux_models)}")
                                logging.error(f"Aux models format error before re-indexing. Type: {type(aux_models)}")
                            else:
                                reindex_success, reindex_msg = reindex_table(
                                    db_conn, selected_table, llm_wrapper, aux_models, qdrant_client
                                )
                                if reindex_success:
                                    st.success(reindex_msg)
                                else:
                                    st.error(reindex_msg)
                                st.rerun() # Refresh overview table
                    elif reindex_disabled:
                        # Optional: Show why disabled
                         st.caption("Re-index disabled: Resources not ready.")


                with col3_manage: # Delete
                    delete_key = f"delete_{selected_table}"
                    confirm_delete_pending_for_this_table = (st.session_state.get("confirm_delete_table") == selected_table)

                    # Show confirmation buttons if confirmation is pending for *this* table
                    if confirm_delete_pending_for_this_table:
                        st.error(f"⚠️ Permanently delete all data and vectors for `{selected_table}`?")
                        if st.button("YES, DELETE", key=f"confirm_delete_yes_{selected_table}", type="primary"):
                            logging.info(f"User confirmed deletion for table: {selected_table}")
                            with st.spinner(f"Deleting '{selected_table}' data..."):
                                delete_success, delete_msg = delete_table_data(db_conn, selected_table, qdrant_client)
                            if delete_success:
                                st.success(delete_msg)
                            else:
                                st.error(delete_msg)

                            # --- Critical: Clear state AFTER action ---
                            st.session_state.confirm_delete_table = None
                            # Don't set selected_table_manage here directly! Let the logic above handle it.

                            # Refresh schema state immediately
                            st.session_state.schema = get_schema_info(db_conn)
                            logging.info(f"Schema refreshed after deletion. Tables: {list(st.session_state.schema.keys())}")

                            # Rerun to update the entire UI, including the selectbox default index
                            st.rerun()

                        if st.button("Cancel Deletion", key=f"confirm_delete_no_{selected_table}"):
                            logging.info(f"User cancelled deletion for table: {selected_table}")
                            st.session_state.confirm_delete_table = None # Clear confirmation flag
                            st.rerun()
                    else:
                        # Show the initial delete button
                        delete_button_disabled = not db_ready # Disable if DB isn't connected
                        if st.button(f"❌ Delete Table Data", key=delete_key, type="secondary", help="Permanently deletes table from SQL database and vector store.", disabled=delete_button_disabled):
                            logging.info(f"Delete button clicked for table: {selected_table}. Requesting confirmation.")
                            # Set state to trigger confirmation on next rerun
                            st.session_state.confirm_delete_table = selected_table
                            st.rerun()


# --- Footer (Optional) ---
st.divider()
st.caption(f"DB Path: {DB_PATH} | Resources OK: Core={core_resources_ok}, Processing={processing_resources_ok}")