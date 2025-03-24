import streamlit as st
import mysql.connector
import openai
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from dotenv import load_dotenv
import google.generativeai as genai
from fpdf import FPDF
import xlsxwriter
import numpy as np
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
import tempfile
import io
import sqlite3

# App title and layout
st.set_page_config(page_title="AI-Powered Text-to-SQL Query Generator for Business Intelligence", layout="wide")

# Load API Keys
openai_env_path = "open_ai_key.env"
gemini_env_path = "gemini_key.env"

load_dotenv(openai_env_path)
load_dotenv(gemini_env_path)

api_key = os.getenv("OPENAI_API_KEY")
gemini_api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    st.error("🚨 OpenAI API key not found. Please check 'open_ai_key.env'.")
    st.stop()

if not gemini_api_key:
    st.error("🚨 Gemini API key not found. Please check 'gemini_key.env'.")
    st.stop()

# Initialize API Clients
client = openai.OpenAI(api_key=api_key)
genai.configure(api_key=gemini_api_key)
model = genai.GenerativeModel("gemini-1.5-pro")

# Initialize OpenAI Embeddings
embeddings = OpenAIEmbeddings(openai_api_key=api_key)

# Initialize session state for dataframes if it doesn't exist
if 'dataframes' not in st.session_state:
    st.session_state.dataframes = {}

# Load data from CSV or Excel
def load_data(uploaded_file):
    try:
        file_extension = uploaded_file.name.split(".")[-1].lower()
        
        if file_extension == "csv":
            df = pd.read_csv(uploaded_file)
        elif file_extension in ["xlsx", "xls"]:
            df = pd.read_excel(uploaded_file)
        else:
            st.error(f"🚨 Unsupported file format for {uploaded_file.name}. Please upload a CSV or Excel file.")
            return None
            
        return df
    except Exception as e:
        st.error(f"🚨 Error loading file {uploaded_file.name}: {e}")
        return None

# Generate schema information from DataFrame
def generate_schema_info(df, table_name):
    schema_info = []
    
    for column in df.columns:
        dtype = str(df[column].dtype)
        sample_values = df[column].dropna().head(3).tolist()
        
        if dtype.startswith("int") or dtype.startswith("float"):
            col_type = "NUMERIC"
        elif dtype.startswith("datetime"):
            col_type = "DATE/DATETIME"
        else:
            col_type = "TEXT"
            
        schema_info.append({
            "column": column,
            "type": col_type,
            "sample_values": sample_values,
            "table": table_name
        })
    
    return schema_info

# Create vector database from DataFrames
def create_vector_database(dataframes):
    # Create documents for vector store
    documents = []
    
    # Add schema information and sample data for each table
    for table_name, df in dataframes.items():
        schema_info = generate_schema_info(df, table_name)
        
        # Add schema information as documents
        schema_text = f"Table name: {table_name}\nColumns:\n"
        for col_info in schema_info:
            schema_text += f"- {col_info['column']} ({col_info['type']}): Sample values: {', '.join(map(str, col_info['sample_values']))}\n"
        
        documents.append(schema_text)
        
        # Add sample records
        for _, row in df.head(5).iterrows():
            row_text = f"Sample record from {table_name}: " + ", ".join([f"{col}: {row[col]}" for col in df.columns])
            documents.append(row_text)
        
        # Add column descriptions with data statistics
        for column in df.columns:
            if pd.api.types.is_numeric_dtype(df[column]):
                stats_text = f"Column {column} in {table_name} statistics: min={df[column].min()}, max={df[column].max()}, mean={df[column].mean():.2f}, median={df[column].median()}"
                documents.append(stats_text)
            elif pd.api.types.is_string_dtype(df[column]):
                unique_vals = df[column].nunique()
                top_vals = df[column].value_counts().head(3).to_dict()
                stats_text = f"Column {column} in {table_name} statistics: {unique_vals} unique values. Top values: {top_vals}"
                documents.append(stats_text)
    
    # Create text splitter
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    docs = [{"content": doc, "metadata": {}} for doc in documents]
    
    # Create vector store
    vector_store = FAISS.from_texts([doc["content"] for doc in docs], embeddings)
    
    return vector_store

# Perform semantic search using the vector database
def semantic_search(vector_store, query, k=3):
    results = vector_store.similarity_search(query, k=k)
    return [doc.page_content for doc in results]

# NLP to SQL using OpenAI GPT-4 with RAG
def openai_nlp_to_sql_with_rag(nlp_query, dataframes, context_docs):
    # Generate schema text for all tables
    schema_text = "Database Schema:\n"
    for table_name, df in dataframes.items():
        schema_text += f"- Table: {table_name} with columns: {', '.join(df.columns.tolist())}\n"
    
    context = "\n".join(context_docs)
    
    prompt = f"""
    Convert the following natural language query into a SQL query using SQLite syntax.

    {schema_text}
    
    Additional context from the database:
    {context}

    Now convert this query:
    User: '{nlp_query}'
    
    Return ONLY the SQL query without any explanations or markdown. The query should work with SQLite syntax.
    """
    try:
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are an expert SQL assistant. You generate only SQL code without explanation."},
                {"role": "user", "content": prompt}
            ]
        )
        sql = response.choices[0].message.content.strip()
        # Clean up the SQL query (remove markdown code blocks if present)
        sql = sql.replace("```sql", "").replace("```", "").strip()
        return sql
    except openai.OpenAIError as e:
        st.error(f"🚨 OpenAI API Error: {e}")
        return None

# NLP to SQL using Gemini with RAG
def gemini_nlp_to_sql_with_rag(nlp_query, dataframes, context_docs):
    # Generate schema text for all tables
    schema_text = "Database Schema:\n"
    for table_name, df in dataframes.items():
        schema_text += f"- Table: {table_name} with columns: {', '.join(df.columns.tolist())}\n"
    
    context = "\n".join(context_docs)
    
    prompt = f"""
    Convert the following natural language query into a SQL query using SQLite syntax.

    {schema_text}
    
    Additional context from the database:
    {context}

    Now convert this query:
    User: '{nlp_query}'
    
    Return ONLY the SQL query without any explanations or markdown.
    """
    try:
        response = model.generate_content(prompt)
        sql = response.text.strip()
        # Clean up the SQL query (remove markdown code blocks if present)
        sql = sql.replace("```sql", "").replace("```", "").strip()
        return sql
    except Exception as e:
        st.error(f"🚨 Gemini API Error: {e}")
        return None

# Execute SQL Query on DataFrames
def execute_sql_query_on_dfs(dataframes, sql_query):
    try:
        # Create a temporary SQLite database in memory
        conn = sqlite3.connect(":memory:")
        
        # Write all dataframes to the database
        for table_name, df in dataframes.items():
            df.to_sql(table_name, conn, index=False, if_exists="replace")
        
        # Execute the query
        st.info(f"🚀 Executing SQL Query: {sql_query}")
        result_df = pd.read_sql_query(sql_query, conn)
        
        conn.close()
        return result_df, result_df.columns
    except Exception as e:
        st.error(f"🚨 SQL Execution Error: {e}")
        return None, None

# Generate Reports (Excel + PDF)
def generate_report(df):
    excel_buffer = io.BytesIO()
    pdf_buffer = io.BytesIO()

    # Generate Excel Report
    with pd.ExcelWriter(excel_buffer, engine="xlsxwriter") as writer:
        df.to_excel(writer, sheet_name="Result", index=False)

    excel_buffer.seek(0)

    # Generate PDF Report
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", style='B', size=16)
    pdf.cell(200, 10, txt="Query Result", ln=True, align='C')
    pdf.ln(10)

    pdf.set_font("Arial", style='B', size=12)
    
    # Adjust column width based on number of columns
    col_width = min(190 / len(df.columns), 30)

    # Headers
    for col in df.columns:
        pdf.cell(col_width, 10, str(col)[:15], border=1, align='C')
    pdf.ln()

    # Data (limit to first 20 rows for PDF)
    pdf.set_font("Arial", size=10)
    for _, row in df.head(20).iterrows():
        for item in row:
            pdf.cell(col_width, 10, str(item)[:15], border=1, align='C')
        pdf.ln()

    # Save visualization for PDF if there's data to visualize
    if not df.empty and len(df.columns) >= 2 and len(df) > 1:
        fig = visualize_data(df, df.columns, save_to_file=True, show_plot=False)
        if fig:
            with tempfile.NamedTemporaryFile(suffix='.png') as tmpfile:
                fig.savefig(tmpfile.name)
                pdf.image(tmpfile.name, x=10, y=None, w=180)
                plt.close(fig)

    pdf.output(dest='S', target=pdf_buffer)
    pdf_buffer.seek(0)
    
    return excel_buffer, pdf_buffer

# Data Visualization
def visualize_data(df, columns, save_to_file=False, show_plot=True):
    if df is None or df.empty or len(df) <= 1:
        return None

    fig, ax = plt.subplots(figsize=(10, 5))

    if len(columns) >= 2:
        # Try to find a numeric column for y-axis
        numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
        if numeric_cols:
            y_col = numeric_cols[0]
            # Find a suitable x column (preferably not numeric)
            non_numeric_cols = [col for col in df.columns if col not in numeric_cols]
            x_col = non_numeric_cols[0] if non_numeric_cols else columns[0]
            if x_col == y_col and len(columns) > 1:
                x_col = columns[1]
            
            # Limit to top 10 values for better visualization
            if len(df) > 10:
                plot_df = df.nlargest(10, y_col) if y_col in df.columns else df.head(10)
            else:
                plot_df = df
                
            # Choose plot type based on data
            if df[x_col].nunique() <= 10:  # Bar chart for fewer unique values
                sns.barplot(x=plot_df[x_col], y=plot_df[y_col], palette="viridis", ax=ax)
                ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
            else:  # Line plot or scatter plot for more values
                sns.scatterplot(x=plot_df[x_col], y=plot_df[y_col], palette="viridis", ax=ax)
            
            ax.set_title(f"{y_col} by {x_col}")
            ax.set_xlabel(x_col)
            ax.set_ylabel(y_col)
            plt.tight_layout()

    if show_plot:
        st.pyplot(fig)

    return fig

# Streamlit UI
st.title("🔄 AI-Powered Text-to-SQL Query Generator for Business Intelligence")
st.subheader("Upload Multiple Data Files and Enter Query")

# Multi-file uploader
uploaded_files = st.file_uploader("📂 Upload CSV or Excel Files", type=["csv", "xlsx", "xls"], accept_multiple_files=True)

if uploaded_files:
    # Process each uploaded file
    for uploaded_file in uploaded_files:
        # Generate a table name from the file name (remove extension and special characters)
        table_name = os.path.splitext(uploaded_file.name)[0]
        table_name = ''.join(c if c.isalnum() else '_' for c in table_name)
        
        # Load the data
        df = load_data(uploaded_file)
        
        if df is not None:
            # Store in session state
            st.session_state.dataframes[table_name] = df
    
    # Check if we have any valid dataframes
    if st.session_state.dataframes:
        st.success(f"✅ {len(st.session_state.dataframes)} file(s) loaded successfully")
        
        # Show data preview for each dataframe
        st.subheader("Data Preview")
        
        # Create tabs for each dataframe
        tabs = st.tabs(list(st.session_state.dataframes.keys()))
        
        for i, (table_name, df) in enumerate(st.session_state.dataframes.items()):
            with tabs[i]:
                st.dataframe(df.head())
                st.text(f"Table: {table_name} - {len(df)} rows, {len(df.columns)} columns")
        
        # Create vector database
        with st.spinner("Creating vector database..."):
            vector_store = create_vector_database(st.session_state.dataframes)
        
        # Query input
        st.subheader("Natural Language Query")
        nlp_query = st.text_area("🔍 Enter your question about the data", "", height=80)
        
        col1, col2 = st.columns([1, 5])
        with col1:
            execute_button = st.button("🚀 Generate & Execute", type="primary")
        with col2:
            st.markdown("Examples: 'Find the top 5 items by sales', 'What's the average age by department?', 'Show monthly revenue trend'")
        
        if execute_button and nlp_query:
            # Perform semantic search
            with st.spinner("Performing semantic search..."):
                context_docs = semantic_search(vector_store, nlp_query)
            
            with st.expander("🔍 Relevant Context from Data"):
                for i, doc in enumerate(context_docs):
                    st.text_area(f"Context {i+1}", doc, height=100)
            
            # Generate SQL with RAG
            with st.spinner("Generating SQL queries..."):
                gpt_sql = openai_nlp_to_sql_with_rag(nlp_query, st.session_state.dataframes, context_docs)
                gemini_sql = gemini_nlp_to_sql_with_rag(nlp_query, st.session_state.dataframes, context_docs)

            # Show generated SQL
            sql_col1, sql_col2 = st.columns(2)
            with sql_col1:
                st.subheader("GPT-4 SQL Query")
                st.code(gpt_sql if gpt_sql else "Failed to generate SQL", language="sql")
                use_gpt = st.button("Use GPT Query") if gpt_sql else False
            
            with sql_col2:
                st.subheader("Gemini SQL Query")
                st.code(gemini_sql if gemini_sql else "Failed to generate SQL", language="sql")
                use_gemini = st.button("Use Gemini Query") if gemini_sql else False
            
            # Custom SQL option
            st.subheader("Or Write Your Own SQL")
            custom_sql = st.text_area("Custom SQL Query", value=gpt_sql if gpt_sql else (gemini_sql if gemini_sql else ""), height=100)
            use_custom = st.button("Run Custom SQL")
            
            # Determine which SQL to run
            sql_to_run = None
            if use_gpt:
                sql_to_run = gpt_sql
            elif use_gemini:
                sql_to_run = gemini_sql
            elif use_custom:
                sql_to_run = custom_sql
            
            # Execute the selected SQL query
            if sql_to_run:
                with st.spinner("Executing SQL query..."):
                    result_df, columns = execute_sql_query_on_dfs(st.session_state.dataframes, sql_to_run)

                if result_df is not None and not result_df.empty:
                    st.subheader("📋 Query Result:")
                    st.dataframe(result_df)
                    
                    st.download_button(
                        "📥 Download Results as CSV",
                        result_df.to_csv(index=False).encode('utf-8'),
                        f"query_result_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        "text/csv"
                    )

                    # Visualization
                    st.subheader("📊 Data Visualization")
                    visualize_data(result_df, columns, save_to_file=False, show_plot=True)
                    
                    # Generate reports for download
                    excel_buffer, pdf_buffer = generate_report(result_df)
                    
                    report_col1, report_col2 = st.columns(2)
                    with report_col1:
                        st.download_button(
                            "📥 Download Excel Report",
                            excel_buffer,
                            file_name=f"query_result_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                            mime="application/vnd.ms-excel"
                        )
                    with report_col2:
                        st.download_button(
                            "📥 Download PDF Report",
                            pdf_buffer,
                            file_name=f"query_result_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                            mime="application/pdf"
                        )
                elif result_df is not None:
                    st.warning("⚠️ Query executed successfully but returned no data.")
    else:
        st.warning("⚠️ No valid files were loaded. Please upload CSV or Excel files.")
else:
    st.info("📤 Please upload one or more CSV or Excel files to begin.")