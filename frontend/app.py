import streamlit as st
import pandas as pd
import sys
from pathlib import Path
import requests

# Add parent directory to path to import agent module
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# import your agent
from backend.main import PurchaseLikelihoodAgent
if "stop_requested" not in st.session_state:
    st.session_state.stop_requested = False

if "has_run" not in st.session_state:
    st.session_state.has_run = False


st.set_page_config(
    page_title="Purchase Prediction Agent | NTU × Enspyre",
    page_icon="frontend/assets/images/enspyre-logo.png"
)

# Streamlit UI
st.image("frontend/assets/images/enspyre-logo.png", width=160)
st.title("Purchase Likelihood Prediction Agent")

st.markdown("""
Upload your survey CSV file and provide product information.  
All processing is **local**; your data stays private.

**Prerequisites:** Make sure Ollama is running (`ollama serve`) and the model `llama3.1` is installed (`ollama pull llama3.1`).
""")

# File upload
uploaded_file = st.file_uploader("Upload your CSV file", type="csv")

# Product information inputs
st.subheader("Product Information")
target_department = st.text_input(
    "Target Department", 
    value="IT",
    help="The department you're targeting (e.g., IT, Finance, HR)"
)

product_description = st.text_area(
    "Product Description",
    placeholder="Describe your product or service. Example: 'Cloud-based CRM software for small businesses'",
    help="This helps the AI understand what you're selling and assess purchase likelihood"
)

# Optional instruction field
instruction = st.text_input(
    "Additional Instructions (Optional)",
    placeholder="Any additional context or instructions",
    help="Optional: Any additional context for the analysis"
)

# Helper function to check if Ollama is running
def check_ollama_connection(base_url: str = "http://localhost:11434", timeout: int = 2):
    """Check if Ollama server is running and accessible."""
    try:
        response = requests.get(f"{base_url}/api/tags", timeout=timeout)
        return response.status_code == 200
    except (requests.exceptions.RequestException, requests.exceptions.Timeout):
        return False

# Process button
if uploaded_file and target_department and product_description:
    # Check Ollama connection before proceeding
    if not check_ollama_connection():
        st.error("""
        **Ollama server is not running!**
        
        Please start Ollama before running predictions:
        1. Make sure Ollama is installed on your system
        2. Start the Ollama server by running: `ollama serve`
        3. Ensure the model 'llama3.1' is available: `ollama pull llama3.1`
        4. Refresh this page and try again
        """)
        st.stop()
    
    if st.button("Run Prediction", type="primary"):
        st.session_state.has_run = True

        # Initialize agent (with caching to avoid re-initialization)
        @st.cache_resource
        def get_agent():
            try:
                return PurchaseLikelihoodAgent()
            except Exception as e:
                st.error(f"Failed to initialize agent: {str(e)}")
                st.error("""
                **Common issues:**
                - Ollama server is not running (run `ollama serve`)
                - Model 'llama3.1' is not installed (run `ollama pull llama3.1`)
                - Network connection issues
                """)
                raise
        
        if st.session_state.has_run:
            try:
                with st.spinner("Initializing agent... This may take a moment."):
                    agent = get_agent()
                
                # Read CSV
                with st.spinner("Reading CSV file..."):
                    df = pd.read_csv(uploaded_file)
                    st.info(f"Loaded {len(df)} rows from CSV file")
                
                # Process with agent
                with st.spinner("Processing... This may take several minutes depending on the number of companies."):
                    response = agent.run_batch_from_csv(
                        df_or_csv=df,
                        instruction=instruction if instruction else "",
                        target_department=target_department,
                        product_description=product_description
                    )
                    
                    st.success("Processing complete!")
                    
                    # Display results
                    st.subheader("Results")
                    st.write(response)
                    
                    # Download button
                    if isinstance(response, pd.DataFrame):
                        csv_data = response.to_csv(index=False, encoding='utf-8-sig')
                        st.download_button(
                            label="Download Results CSV",
                            data=csv_data,
                            file_name="prediction_results.csv",
                            mime="text/csv"
                        )
                        
                        # Show summary statistics
                        if 'purchase_likelihood_score' in response.columns:
                            st.subheader("Summary Statistics")
                            col1, col2, col3 = st.columns(3)
                            
                            with col1:
                                st.metric(
                                    "Companies Analyzed",
                                    len(response)
                                )
                            
                            with col2:
                                avg_score = response['purchase_likelihood_score'].mean()
                                st.metric(
                                    "Average Purchase Likelihood",
                                    f"{avg_score:.1f}%"
                                )
                            
                            with col3:
                                high_prob = len(response[response['purchase_likelihood_score'] >= 70])
                                st.metric(
                                    "High Probability (≥70%)",
                                    high_prob
                                )
            except requests.exceptions.ConnectionError:
                st.error("""
                **Connection Error:**
                Cannot connect to Ollama server at http://localhost:11434
                
                Please ensure:
                1. Ollama is installed and running (`ollama serve`)
                2. The server is accessible at http://localhost:11434
                3. No firewall is blocking the connection
                """)
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
                st.exception(e)
        
        # STEP 4 — DISPLAY RESULTS (NO COMPUTATION)
        if "response" in st.session_state:
            st.success("Processing complete!")
            st.subheader("Results")
            st.write(st.session_state.response)

            if isinstance(st.session_state.response, pd.DataFrame):
                csv_data = st.session_state.response.to_csv(index=False, encoding="utf-8-sig")
                st.download_button(
                    "Download Results CSV",
                    csv_data,
                    "prediction_results.csv",
                    "text/csv"
                )

elif uploaded_file:
    st.info("Please fill in the Target Department and Product Description fields to proceed.")
else:
    st.info("Please upload a CSV file to get started.")

st.caption(
    "Developed by NTU students as part of the Big Data & Agentic AI course, in collaboration with Enspyre."
)