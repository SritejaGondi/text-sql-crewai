import os
import re
import tempfile
import streamlit as st
import pandas as pd
import json  # For formatting JSON output
from sqlalchemy import create_engine, MetaData, text
from textwrap import dedent

# --- Streamlit Page Setup: Must be the first Streamlit command ---
st.set_page_config(page_title="Text-to-SQL & Python Coding Agent", layout="wide")
st.title("Text-to-SQL & Python Coding Agent")

# --- Custom CSS for Light Purple & White Theme ---
st.markdown(
    """
    <style>
    /* Main background and text colors */
    html, body, [data-testid="stAppViewContainer"] {
        background-color: #FFFFFF !important;
        color: #4B0082 !important;
    }
    [data-testid="stHeader"] .css-1cpxqw2 {
        color: #FFFFFF !important;
    }
    /* Buttons styling */
    .stButton>button {
        background-color: #E6E6FA !important;
        color: #4B0082 !important;
        border: none !important;
    }
    /* Text input styling */
    .stTextInput>div>div>input {
        background-color: #f9f9f9 !important;
        color: #4B0082 !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# --- LangChain & CrewAI Imports ---
from langchain_community.utilities.sql_database import SQLDatabase
from langchain_community.agent_toolkits.sql.toolkit import SQLDatabaseToolkit
from langchain_community.tools.sql_database.tool import (
    InfoSQLDatabaseTool,
    ListSQLDatabaseTool,
    QuerySQLCheckerTool,
    QuerySQLDatabaseTool,
)
from crewai import Agent, Task, Crew, Process
from crewai.tools import tool
from langchain.chat_models import init_chat_model
from crewai import LLM
# from crewai.knowledge.source.csv_knowledge_source import CSVKnowledgeSource
from crewai_tools import CodeInterpreterTool

# --- Database Source Selection (for SQL Agent) ---
st.markdown("### Database Source Selection")
db_source_option = st.radio("Select database source:", ["CSV Files", "SQLite DB File"], key="db_source")

if db_source_option == "CSV Files":
    uploaded_csv_files = st.file_uploader("Upload CSV files", type="csv", accept_multiple_files=True, key="csv_upload")
else:
    uploaded_db_file = st.file_uploader("Upload a SQLite DB file", type=["db"], key="db_upload")

if st.button("Load Database", key="load_db"):
    try:
        if db_source_option == "CSV Files" and not uploaded_csv_files:
            st.error("No CSV files uploaded!")
            st.stop()
            
        temp_db_path = None

        if db_source_option == "CSV Files":
            temp_db = tempfile.NamedTemporaryFile(delete=False, suffix=".db")
            temp_db_path = temp_db.name
            temp_db.close()

            engine = create_engine(f"sqlite:///{temp_db_path}")
            metadata_obj = MetaData()

            for csv_file in uploaded_csv_files:
                df = pd.read_csv(csv_file)
                table_name = os.path.splitext(csv_file.name)[0]
                df.to_sql(table_name, engine, index=False, if_exists='replace')
                st.write(f"Table '{table_name}' created successfully.")
            
            # Reflect schema for CSV-based temp DB
            metadata_obj.reflect(bind=engine)
        
        else:  # SQLite DB Upload Case
            if uploaded_db_file is None:
                st.error("Please upload a SQLite DB file!")
                st.stop()

            temp_db = tempfile.NamedTemporaryFile(delete=False, suffix=".db")
            temp_db_path = temp_db.name

            with open(temp_db_path, "wb") as f:
                f.write(uploaded_db_file.getvalue())

            engine = create_engine(f"sqlite:///{temp_db_path}")
            metadata_obj = MetaData()
            metadata_obj.reflect(bind=engine)
            st.success(f"Connected to database: {temp_db_path}")

        # Extract column names for each table and store in a dictionary.
        db_columns = {}
        for table_name, table in metadata_obj.tables.items():
            db_columns[table_name] = list(table.columns.keys())
        st.write("Extracted Database Columns:", db_columns)
        
        st.session_state.update({
            "engine": engine, 
            "metadata_obj": metadata_obj, 
            "db_loaded": True,
            "db_columns": db_columns  # Save columns for later use by the coding agent.
        })
    
    except Exception as e:
        st.error(f"Error loading the database: {e}")
        st.stop()


# --- LLM Configuration (used by both agents) ---
st.header("LLM Configuration (Together AI)")
together_api_key = st.text_input("Enter your Together AI API key", type="password", key="together_api_key")
together_model = st.text_input("Enter Together AI Model Name", value="meta-llama/Llama-3.3-70B-Instruct-Turbo-Free", key="together_model")

if st.button("Configure Model", key="configure_model"):
    if not together_api_key or not together_model:
        st.error("Please provide both API key and model name!")
    else:
        os.environ["TOGETHER_API_KEY"] = together_api_key
        crewllm = LLM(model=f"together_ai/{together_model}", streaming=True, temperature=0.7)
        lcllm = init_chat_model(together_model, streaming=True, temperature=0.7, model_provider="together")
        st.session_state["crewllm"] = crewllm
        st.session_state["lcllm"] = lcllm
        st.success(f"Together AI LLM configured: {together_model}")

# --- Create Tabs for SQL Agent and Coding Agent ---
tabs = st.tabs(["SQL Query Agent", "Python Coding Agent"])

# ===================== SQL Query Agent Tab =====================
with tabs[0]:
    st.header("SQL Query Agent")
    # Display DB schema in sidebar (only if DB is loaded)
    if st.session_state.get("db_loaded") and "lcllm" in st.session_state:
        with st.sidebar:
            st.subheader("Database Schema")
            metadata_obj = st.session_state["metadata_obj"]
            for table_name, table in metadata_obj.tables.items():
                st.markdown(f"**{table_name}**")
                st.write(list(table.columns.keys()))
    
        # --- Define Crew AI Tools for SQL Agent ---
        engine = st.session_state["engine"]
        db = SQLDatabase(engine)
        lcllm = st.session_state["lcllm"]
        crewllm = st.session_state["crewllm"]

        toolkit = SQLDatabaseToolkit(db=db, llm=lcllm)
        list_tool = ListSQLDatabaseTool(db=db, llm=lcllm)
        info_tool = InfoSQLDatabaseTool(db=db, llm=lcllm)
        checker_tool = QuerySQLCheckerTool(db=db, llm=lcllm)
        query_tool = QuerySQLDatabaseTool(db=db, llm=lcllm)

        @tool("list_tables")
        def list_tables():
            """Retrieve and return a list of all tables in the connected database."""
            return db.get_table_names()

        @tool("tables_schema")
        def tables_schema(table_names: str):
            """Fetch and return the schema details of the specified tables."""
            return info_tool.run(table_names)

        @tool("check_sql")
        def check_sql(query: str):
            """Validate the correctness of a given SQL query before execution."""
            return checker_tool.run(query)

        @tool("execute_sql")
        def execute_sql(query: str):
            """Execute the provided SQL query and return the results."""
            return query_tool.run(query)
        
        # --- Define Crew AI Agents for SQL Agent ---
        understanding_agent = Agent(
            role="Semantic Analysis Expert",
            goal="Understand the user's question: {query}; and determine what the user is asking based on world knowledge.",
            backstory="A retail/supermarket expert who knows details about products and orders in the database.",
            llm=crewllm,
            verbose=True
        )

        schema_analysis_agent = Agent(
            role="Database Schema Expert",
            goal="Identify which tables in the database contain information relevant to the user’s query.",
            backstory="An expert SQL database agent that maps user queries to correct database structures.",
            llm=crewllm,
            tools=[list_tables, tables_schema],
            verbose=True
        )

        sql_query_agent = Agent(
            role="SQL Query Specialist",
            goal="Construct syntactically correct SQL queries based on the previous agents’ findings.",
            backstory="An expert SQL agent that writes SQL queries to provide insights. Returns ONLY the SQL query. DOES NOT generate any DML statements (INSERT, UPDATE, DELETE, DROP, etc.)",
            llm=crewllm,
            tools=[check_sql],
            verbose=True
        )

        # --- Define Crew AI Tasks for SQL Agent ---
        understanding_task = Task(
            description="Analyze the user query to understand context and meaning.", 
            expected_output="A breakdown analysis of the user question.", 
            agent=understanding_agent
        )

        schema_task = Task(
            description="Analyze database schema and map it to relevant columns/tables based on the user prompt.", 
            expected_output="A mapping of tables and columns relevant to the query.", 
            agent=schema_analysis_agent, 
            context=[understanding_task]
        )

        sql_task = Task(
            description="Construct and execute an SQL query using schema analysis results.", 
            expected_output="The executed SQL query and its results.", 
            agent=sql_query_agent, 
            context=[understanding_task, schema_task]
        )

        # --- Create and Execute the SQL Crew ---
        crew = Crew(
            agents=[understanding_agent, schema_analysis_agent, sql_query_agent],
            tasks=[understanding_task, schema_task, sql_task],
            process=Process.sequential,
            verbose=True
        )

        # --- Overview Sections ---
        st.markdown("## Agents & Tasks Overview")
        st.markdown("### Agents")
        st.write("- **Understanding Agent**: Interprets the user's question.")
        st.write("- **Schema Analysis Agent**: Determines which tables/columns are relevant.")
        st.write("- **SQL Query Agent**: Builds the SQL query.")
    
        st.markdown("### Tasks")
        st.write("1. **Understanding Task**: Break down the user's question.")
        st.write("2. **Schema Task**: Identify relevant tables/columns.")
        st.write("3. **SQL Task**: Write the SQL query.")
    
        st.markdown("## Tools in Use")
        st.write("- **list_tables**: Retrieve all table names.")
        st.write("- **tables_schema**: Get schema info for specific tables.")
        st.write("- **check_sql**: Validate SQL queries.")

        # --- SQL Agent Query Execution UI ---
        user_query = st.text_input("Ask a SQL-related question:", key="sql_prompt")

        if st.button("Run Query", key="run_query"):
            if not user_query.strip():
                st.error("Please enter a query!")
            else:
                st.info("Processing your query, please wait...")
                crew_output = crew.kickoff(inputs={"query": user_query})
                st.success("Crew query execution complete!")
                
                st.markdown("### Predicted SQL Query")
                predicted_sql = crew_output.raw.strip()
                st.code(predicted_sql, language="sql")
                
                # Optionally, execute the predicted SQL query (if desired)
                try:
                    # Get the third task output
                    raw_sql_output = predicted_sql
                    # Remove code block formatting (e.g., ```sql and ```)
                    cleaned_sql = re.sub(r"```sql\s*", "", raw_sql_output)
                    cleaned_sql = re.sub(r"\s*```", "", cleaned_sql).strip()
                    print(cleaned_sql)
                    
                    with engine.connect() as conn:
                        stmt = text(cleaned_sql)
                        query_results = conn.execute(stmt)
                        columns = query_results.keys()
                        rows = query_results.fetchall()
                        
                        if not rows:
                            st.error("The predicted SQL query did not return an output.")
                        else:
                            df = pd.DataFrame(rows, columns=columns)
                            st.write("**Query Results:**")
                            st.dataframe(df)
                except Exception as ex:
                    st.error(f"Error executing the predicted SQL query: {ex}")
                
                st.markdown("### Token Usage")
                st.write(crew_output.token_usage)
                
                st.markdown("### Task Outputs")
                st.write_stream(crew_output.tasks_output)
    else:
        st.sidebar.subheader("Database Schema")
        st.sidebar.write("No database loaded yet or LLM not configured.")
        st.write("Please load a database and configure the LLM to proceed.")

# ===================== Python Coding Agent Tab =====================
with tabs[1]:
    st.header("Python Agent")
    st.markdown("Ask a python question. The agent will write python code to answer your question.")
    # --- Overview Sections ---
    st.markdown("## Agents & Tasks Overview")
    st.markdown("### Agents")
    st.write("- **Coding Agent**: Writes Python code based on user prompt.")
    
    st.markdown("### Tasks")
    st.write("""**Coding Task**: Create a new csv file by merging the products.csv, aisles.csv and departments.csv files. In this new csv
file, add the following product feature columns: contains_dairy, contains_meat, contains_fish, contains_gluten, contains_sugar,
contains_nuts, contains_alcohol, contains_caffeine, contains_vegetables, contains_fruit, contains_liquid. In these new columns,
classify each of the products using '1' for yes and '0' for no based on your understanding of retail supermarket products and grocery items. 

Define a function that takes each row in the merged csv file and updates ingredient flags (like contains_dairy, contains_meat,
contains_fish, contains_gluten, etc.) using common keywords found in product names. 

Use logical keyword checks for each product name (e.g., "milk", "cheese" for dairy; "beef", "chicken", "pork", "lamb" for meat;
"bread", "flour" for gluten, etc.), and ensure that more specific phrases like "gluten free" or "sugar free" override the flags if
present.""")
    
    st.markdown("## Tools in Use")
    st.write("- **Code Interpreter**: Writing and executing Python 3 code.")
    
    coding_query = st.text_input("Enter your python question:", key="coding_query")
    
    if st.button("Run Coding Agent", key="run_coding"):
        if not coding_query.strip():
            st.warning("Please enter a valid question!")
        else:
            if "crewllm" not in st.session_state:
                st.error("LLM not configured. Please configure the LLM first.")
            else:
                st.info("Processing your analysis query, please wait...")
                # Use the same crewllm from session_state for the coding agent.
                crewllm = st.session_state["crewllm"]

                # Retrieve the db_columns context (if available)
                db_columns = st.session_state.get("db_columns", {})
                # Convert db_columns to a JSON string and escape curly braces.
                db_columns_str = json.dumps(db_columns, indent=2)
                db_columns_str_escaped = db_columns_str.replace("{", "{{").replace("}", "}}")
                
                # Assuming you have defined or imported CodeInterpreterTool somewhere:
                code_interpreter = CodeInterpreterTool()

                # Create a coding agent with code execution enabled.
                coding_agent = Agent(
                    role="Python Data Transformation Expert",
                    goal="Analyze data and provide insights using Python code.",
                    backstory="You are a senior python developer with extensive experience in python, data science, statistics and probability and business analytics.",
                    llm=crewllm,
                    # allow_code_execution=True,
                    # code_execution_mode="safe",  # Uses Docker for safety
                    # max_execution_time=300,  # 5-minute timeout
                    # max_retry_limit=3,  # More retries for complex code tasks
                    tools=[code_interpreter],
                    verbose=True
                )
                
                # Create a task that instructs the agent to write and execute code.
                task_description = f"""Create a new csv file by merging the products.csv, aisles.csv and departments.csv files. In this new csv
file, add the following product feature columns: contains_dairy, contains_meat, contains_fish, contains_gluten, contains_sugar,
contains_nuts, contains_alcohol, contains_caffeine, contains_vegetables, contains_fruit, contains_liquid. In these new columns,
classify each of the products using '1' for yes and '0' for no based on your understanding of retail supermarket products and grocery items.

Define a function that takes each row in the merged csv file and updates ingredient flags (like contains_dairy, contains_meat,
contains_fish, contains_gluten, etc.) using common keywords found in product names.

Use logical keyword checks for each product name (e.g., "milk", "cheese" for dairy; "beef", "chicken", "pork", "lamb" for meat;
"bread", "flour" for gluten, etc.), and ensure that more specific phrases like "gluten free" or "sugar free" override the flags if
present.

Here are the column names of the csv files: {db_columns_str_escaped}"""
                
                data_analysis_task = Task(
                    description=task_description,
                    expected_output="The complete Python code with ALL the data transformation requirements.",
                    agent=coding_agent,
                    context=[schema_task]  # Optional: include additional context if available.
                )

                # Create a crew with the coding agent.
                coding_crew = Crew(
                    agents=[coding_agent],
                    tasks=[data_analysis_task],
                    process=Process.sequential,
                    verbose=True
                )
                # Pass the query as input to the crew.
                coding_result = coding_crew.kickoff(inputs={"query": coding_query})
                st.success("Coding Agent execution complete!")
                
                st.markdown("### Coding Agent Output")
                st.code(coding_result.raw, language="python")
                
                st.markdown("### Token Usage")
                st.write(coding_result.token_usage)
                
                st.markdown("### Task Outputs")
                st.write_stream(coding_result.tasks_output)

