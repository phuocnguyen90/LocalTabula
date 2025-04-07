# app.py
import streamlit as st
import pandas as pd
import os
import time
import logging

# Import functions from utils.py
from utils import (
    setup_environment, get_db_connection, init_qdrant_client, get_llm_wrapper,
    get_cached_aux_models, get_schema_info, process_uploaded_data,
    process_natural_language_query, table_exists,
    # Import new management functions
    get_sqlite_table_row_count, get_qdrant_collection_info,
    reindex_table, delete_table_data,
    derive_requirements_from_history,
    QDRANT_COLLECTION_PREFIX  # Import prefix constant
)

# --- Config Logging & Filter ---
# ... (Logging setup including TorchClassesPathFilter as before) ...


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


# --- Perform Readiness Checks ---
print(f"DEBUG [app.py Readiness Check]: Type of aux_models before status check: {type(aux_models)}")
aux_models_ready = isinstance(aux_models, dict) and aux_models.get("status") == "loaded"
db_ready = db_conn is not None
qdrant_ready = qdrant_client is not None
llm_ready = llm_wrapper and llm_wrapper.is_ready
aux_models_ready = aux_models and aux_models.get("status") == "loaded"

# Processing requires DB, Qdrant, Embedder(Aux), optionally LLM for column suggestions
processing_resources_ok = db_ready and qdrant_ready and aux_models_ready  # Add llm_ready if needed
# Chat requires DB, Qdrant, LLM, Aux (SQL/Embedder)
core_resources_ok = db_ready and qdrant_ready and llm_ready and aux_models_ready

# --- Initialize Session State ---
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Hi! Load data via sidebar, then ask questions."}]
if "schema" not in st.session_state:
    st.session_state.schema = get_schema_info(db_conn)
if "confirm_replace_needed" not in st.session_state:
    st.session_state.confirm_replace_needed = False
if "confirm_replace_details" not in st.session_state:
    st.session_state.confirm_replace_details = {}
if "process_now_confirmed" not in st.session_state:
    st.session_state.process_now_confirmed = False
# Add state for management tab
if "selected_table_manage" not in st.session_state:
    st.session_state.selected_table_manage = None
if "confirm_delete_table" not in st.session_state:
    st.session_state.confirm_delete_table = None  # Store table name to delete

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
                    schema=st.session_state.schema,
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
        gsheet_published_url = st.text_input("Paste Published Google Sheet CSV URL", help="Use File > Share > Publish to web > CSV format", key="data_gsheet")
        table_name_input = st.text_input("Enter Table Name for Database", help="Replaces existing table if name conflicts.", key="data_table_name")

        # Confirmation logic for replacing table
        confirm_needed = st.session_state.get("confirm_replace_needed", False)
        confirm_details = st.session_state.get("confirm_replace_details", {})

        # Display confirmation buttons if needed for the *current* input table name
        if confirm_needed and confirm_details.get("table_name") == table_name_input and table_name_input:
            st.warning(f"Table `{table_name_input}` already exists.")
            col1, col2 = st.columns(2)
            with col1:
                if st.button("Yes, Replace Table", key="confirm_replace_yes"):
                    st.session_state.confirm_replace_needed = False
                    st.session_state.process_now_confirmed = True  # Signal to process
                    st.rerun()
            with col2:
                if st.button("No, Cancel Load", key="confirm_replace_no"):
                    st.session_state.confirm_replace_needed = False
                    st.session_state.confirm_replace_details = {}
                    st.info("Load operation cancelled.")
                    st.rerun()
        else:
            # Show primary load button
            load_disabled = not processing_resources_ok or not table_name_input or (not uploaded_file and not gsheet_published_url)
            if st.button("Load and Process Data", disabled=load_disabled, key="process_data"):
                st.session_state.process_now_confirmed = True  # Signal to process (even if no confirmation was shown)
                st.rerun()

        # Processing Logic (runs after button click + potential confirmation rerun)
        if st.session_state.get("process_now_confirmed", False):
            st.session_state.process_now_confirmed = False  # Consume flag
            current_table_name = table_name_input  # Use name from input field
            needs_confirm_now = db_ready and table_exists(conn=db_conn, table_name=current_table_name)

            if needs_confirm_now and confirm_details.get("table_name") != current_table_name:
                # If confirmation needed but wasn't shown for this table, show it now
                st.session_state.confirm_replace_needed = True
                st.session_state.confirm_replace_details = {"table_name": current_table_name}
                st.rerun()
            elif processing_resources_ok and (not needs_confirm_now or confirm_details.get("table_name") == current_table_name):
                # Proceed if resources OK AND (table doesn't exist OR confirmation was for this table)
                with st.spinner("Processing data..."):
                    # --- Add Type Check Before Call ---
                    print(f"DEBUG [app.py Process Data]: Type of aux_models before passing: {type(aux_models)}")
                    if not isinstance(aux_models, dict): print("ERROR: aux_models IS STRING before process_uploaded_data!")
                    # Pass llm_wrapper for column suggestion
                    success, message = process_uploaded_data(
                        uploaded_file, gsheet_published_url, current_table_name,
                        db_conn, llm_wrapper, aux_models, qdrant_client,
                        replace_confirmed=True  # Confirmation handled by UI flow
                    )
                if message:
                    st.info(f"Processing Log:\n```\n{message}\n```")
                if success:
                    st.session_state.schema = get_schema_info(db_conn)  # Refresh schema
                    st.success(f"Processing completed for `{current_table_name}`.")
                    st.session_state.confirm_replace_needed = False  # Clear state
                    st.session_state.confirm_replace_details = {}
                    st.rerun()  # Rerun to update UI
                else:
                    st.error(f"Processing failed for `{current_table_name}`.")
                    st.session_state.confirm_replace_needed = False  # Clear state even on fail
                    st.session_state.confirm_replace_details = {}

    st.markdown("---")

    # --- Section 2: Database Overview ---
    st.subheader("Database Status & Overview")
    current_schema = st.session_state.schema
    if not db_ready:
        st.warning("Database not connected.")
    elif not current_schema:
        st.info("No tables found in the database.")
    else:
        st.write(f"**Tables found:** {len(current_schema)}")
        # Prepare data for display table
        overview_data = []
        for table_name, columns in current_schema.items():
            row_count = get_sqlite_table_row_count(db_conn, table_name)
            qdrant_collection_name = f"{QDRANT_COLLECTION_PREFIX}{table_name}"
            qdrant_info = get_qdrant_collection_info(qdrant_client, qdrant_collection_name)
            overview_data.append({
                "Table Name": table_name,
                "Columns": len(columns),
                "SQLite Rows": row_count if row_count is not None else "N/A",
                "Vector Collection": qdrant_collection_name if qdrant_info else "Not Found",
                "Vector Points": qdrant_info.get("points_count") if qdrant_info else "N/A",
                "Vector Dim": qdrant_info.get("vector_size") if qdrant_info else "N/A"
            })
        st.dataframe(pd.DataFrame(overview_data), use_container_width=True)

        # Detailed view per table
        with st.expander("Show Table Details"):
            for table_name, columns in current_schema.items():
                st.markdown(f"**Table: `{table_name}`**")
                st.write("Columns:", ", ".join([f"`{col}`" for col in columns]))

    st.markdown("---")

    # --- Section 3: Manage Existing Tables ---
    st.subheader("Manage Existing Tables")
    if not current_schema:
        st.info("Load data first to manage tables.")
    else:
        table_options = list(current_schema.keys())
        # Use session state to preserve selection across reruns
        if st.session_state.selected_table_manage not in table_options:
            st.session_state.selected_table_manage = table_options[0]  # Default to first

        selected_table = st.selectbox(
            "Select Table to Manage:",
            options=table_options,
            key="selected_table_manage"  # Use key to bind to session state
        )

        if selected_table:
            st.markdown(f"**Actions for table: `{selected_table}`**")
            col1, col2, col3 = st.columns(3)

            with col1:  # View Data
                if st.button(f"View Sample Data", key=f"view_{selected_table}"):
                    try:
                        sample_df = pd.read_sql_query(f"SELECT * FROM {selected_table} LIMIT 10", db_conn)
                        st.dataframe(sample_df, use_container_width=True)
                    except Exception as e:
                        st.error(f"Failed to read sample data: {e}")

            with col2:  # Re-index
                if st.button(f"🔄 Re-index Vectors", key=f"reindex_{selected_table}", help="Re-calculates and replaces vector embeddings for this table."):
                    if processing_resources_ok and llm_ready:
                        with st.spinner(f"Re-indexing '{selected_table}'... This may take time."):
                            # --- Add Type Check Before Call ---
                            print(f"DEBUG [app.py Process Data]: Type of aux_models before passing: {type(aux_models)}")
                            if not isinstance(aux_models, dict): print("ERROR: aux_models IS STRING before process_uploaded_data!")
                            reindex_success, reindex_msg = reindex_table(
                                db_conn, selected_table, llm_wrapper, aux_models, qdrant_client
                            )
                        if reindex_success:
                            st.success(reindex_msg)
                        else:
                            st.error(reindex_msg)
                        st.rerun()  # Refresh overview
                    else:
                        st.error("Cannot re-index: Core resources (DB, Qdrant, Models, LLM) not ready.")

            with col3:  # Delete
                delete_key = f"delete_{selected_table}"
                # Confirmation logic for delete
                if st.session_state.get("confirm_delete_table") == selected_table:
                    st.error(f"⚠️ Confirm Deletion of `{selected_table}`?")
                    if st.button("YES, DELETE (Permanent)", key=f"confirm_delete_yes_{selected_table}", type="primary"):
                        with st.spinner(f"Deleting '{selected_table}' data..."):
                            delete_success, delete_msg = delete_table_data(db_conn, selected_table, qdrant_client)
                        if delete_success:
                            st.success(delete_msg)
                        else:
                            st.error(delete_msg)
                        # Clear state and refresh
                        st.session_state.confirm_delete_table = None
                        st.session_state.selected_table_manage = None  # Reset selection
                        st.session_state.schema = get_schema_info(db_conn)  # Update schema
                        st.rerun()
                    if st.button("Cancel Deletion", key=f"confirm_delete_no_{selected_table}"):
                        st.session_state.confirm_delete_table = None
                        st.rerun()
                else:
                    if st.button(f"❌ Delete Table Data", key=delete_key, type="secondary", help="Permanently deletes table from SQL and vector data."):
                        # Set state to trigger confirmation on next rerun
                        st.session_state.confirm_delete_table = selected_table
                        st.rerun()

# --- Footer (Optional) ---
# st.caption("...")
