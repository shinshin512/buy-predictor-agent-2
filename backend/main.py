from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_ollama import ChatOllama
from langchain.prompts import PromptTemplate
from typing import Dict, Any, Optional, List
import pandas as pd
import json
import os
import sys
from backend.config import AGENT_PROMPT

# Add survey_feature_engineering to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'survey_feature_engineering'))
from backend.tools.data_preparation.data_preparation import engineer_features_from_csv


class PurchaseLikelihoodAgent:
    """
    LangChain Agent that predicts purchase likelihood from survey data.
    No longer uses ML model tool - calculates final score directly.
    """
    
    def __init__(self, model_name: str = "llama3.1", base_url: str = "http://localhost:11434", 
                 num_gpu: int = 1, num_ctx: int = 4096):
        """
        Initialize the agent with necessary components.
        
        Args:
            model_name: Name of the Ollama model to use (default: "llama3.1")
            base_url: Ollama server URL (default: "http://localhost:11434")
            num_gpu: Number of GPU layers to use (0 = CPU only, 1 = use GPU if available)
            num_ctx: Context window size (increased to 4096 for better reasoning)
        """
        # Initialize the LLM with Ollama
        self.llm = ChatOllama(
            model=model_name,
            base_url=base_url,
            temperature=0,
            num_gpu=num_gpu,
            num_ctx=num_ctx,
            format="",
        )
        
        print(f"✓ Initialized agent with model: {model_name}")
        print(f"  GPU layers: {num_gpu} (0=CPU only, 1=GPU if available)")
        print(f"  Context size: {num_ctx} tokens")
        
        # No tools needed anymore
        self.tools = []
        
        # Create prompt
        self.prompt = PromptTemplate.from_template(AGENT_PROMPT)
        
        # Create agent
        agent = create_tool_calling_agent(
            llm=self.llm,
            tools=self.tools,
            prompt=self.prompt
        )
        
        # Create agent executor
        self.agent_executor = AgentExecutor(
            agent=agent,
            tools=self.tools,
            verbose=True,
            handle_parsing_errors=True,
            max_iterations=1,
            early_stopping_method="force",
            return_intermediate_steps=True
        )
    
    def predict(self, target_department: str, product_description: str, raw_dataset: Dict[str, Any]) -> Dict[str, Any]:
        """
        Predict purchase likelihood for a company.
        
        Args:
            target_department: The target department for the product
            product_description: Description of the product being sold.
            raw_dataset: Dictionary containing company info and survey Q&A
            
        Returns:
            Dictionary with prediction score and details
        """
        # Format the dataset for the agent
        dataset_str = json.dumps(raw_dataset, indent=2)
        
        # Run the agent
        result = self.agent_executor.invoke({
            "input": f"Predict the purchase likelihood score for this company.",
            "target_department": target_department,
            "product_description": product_description,
            "raw_dataset": dataset_str
        })
        
        # Parse and return the result from output
        parsed_result = self._parse_result(result["output"])
        return parsed_result
    
    def _parse_result(self, output: str) -> Dict[str, Any]:
        """Parse the agent's output to extract the final scores."""
        try:
            # Look for JSON in the output
            if "{" in output and "}" in output:
                json_start = output.find("{")
                json_end = output.rfind("}") + 1
                json_str = output[json_start:json_end]
                parsed = json.loads(json_str)
                
                # Ensure all required fields exist
                if "purchase_likelihood_score" not in parsed:
                    parsed["purchase_likelihood_score"] = "N/A"
                if "bant_score" not in parsed:
                    parsed["bant_score"] = "N/A"
                if "sentiment_score" not in parsed:
                    parsed["sentiment_score"] = "N/A"
                    
                return parsed
            else:
                # Fallback if no JSON found
                return {
                    "purchase_likelihood_score": "N/A",
                    "bant_score": "N/A",
                    "sentiment_score": "N/A",
                    "raw_output": output
                }
        except Exception as e:
            return {
                "error": f"Failed to parse output: {str(e)}",
                "purchase_likelihood_score": "N/A",
                "bant_score": "N/A",
                "sentiment_score": "N/A",
                "raw_output": output
            }
    
    def run(
        self,
        raw_dataset: Dict[str, Any],
        target_department: str,
        product_description: str,
    ) -> Dict[str, Any]:
        """
        Single-company execution.
        """
        return self.predict(
            target_department=target_department,
            product_description=product_description,
            raw_dataset=raw_dataset
        )

    
    def run_batch_from_csv(
        self,
        df_or_csv,
        target_department: str,
        product_description: str,
        instruction: str = "",
        stop_flag=None
    ) -> pd.DataFrame:
        """
        Streamlit-friendly batch runner.
        """

        # Load data
        if isinstance(df_or_csv, pd.DataFrame):
            features_df = engineer_features_from_csv(df_or_csv)
        else:
            features_df = engineer_features_from_csv(df_or_csv)

        datasets = convert_engineered_features_to_agent_format(features_df)

        results = []

        for i, dataset in enumerate(datasets, 1):
            if stop_flag and stop_flag():
                break

            result = self.run(
                raw_dataset=dataset,
                target_department=target_department,
                product_description=product_description,
            )

            row = features_df.iloc[i - 1].to_dict()

            results.append({
                **row,
                "purchase_likelihood_score": result.get("purchase_likelihood_score"),
                "bant_score": result.get("bant_score"),
                "sentiment_score": result.get("sentiment_score"),
            })

        return pd.DataFrame(results)


def convert_engineered_features_to_agent_format(features_df: pd.DataFrame) -> List[Dict[str, Any]]:
    """
    Convert engineered features DataFrame to the format expected by the agent.
    
    Args:
        features_df: DataFrame with engineered company-level features
        
    Returns:
        List of dictionaries, each containing structured data for one company
    """
    datasets = []
    
    # Identify company info columns (non-question, non-engineered feature columns)
    company_info_cols = [
        'company_id', 'survey_name', 'company_size', 'company_size_category',
        'capital_millions', 'sic_code', 'industry'
    ]
    
    # Question columns start with 'Q_'
    question_cols = [col for col in features_df.columns if col.startswith('Q_')]
    
    for idx, row in features_df.iterrows():
        # Extract company info
        company_info = {}
        for col in company_info_cols:
            if col in features_df.columns:
                val = row[col]
                # Convert to appropriate type
                if pd.notna(val):
                    company_info[col] = val
                else:
                    company_info[col] = None
        
        # Add company_id as company_name if available
        if 'company_id' in company_info:
            company_info['company_name'] = str(company_info['company_id'])
        
        # Extract survey responses (question columns and other engineered features)
        survey_responses = {}
        
        # Add all question columns
        for col in question_cols:
            val = row[col]
            if pd.notna(val):
                survey_responses[col] = str(val)
        
        # Add other relevant engineered features as context
        context_features = [
            'total_calls', 'completed_survey', 'has_survey_responses',
            'max_questions_answered', 'avg_questions_answered',
            'contacted_it_dept', 'contacted_manager', 'contacted_director',
            'response_rate', 'has_email', 'has_work_email'
        ]
        for col in context_features:
            if col in features_df.columns:
                val = row[col]
                if pd.notna(val):
                    survey_responses[f"feature_{col}"] = val
        
        # Combine into structured dataset
        dataset = {
            "row_index": idx,
            "company_info": company_info,
            "survey_responses": survey_responses
        }
        
        datasets.append(dataset)
    
    return datasets


def batch_predict_from_csv(
    agent: PurchaseLikelihoodAgent,
    csv_path: str,
    target_department: str,
    product_description: str,
    company_info_columns: Optional[List[str]] = None,
    output_csv_path: Optional[str] = None
) -> pd.DataFrame:
    """
    Load CSV, engineer features, predict purchase likelihood for each company, and optionally save results.
    
    Args:
        agent: Initialized PurchaseLikelihoodAgent instance
        csv_path: Path to input CSV file
        target_department: Target department for the product
        product_description: Description of the product being sold.
        company_info_columns: (Deprecated - kept for compatibility, not used)
        output_csv_path: Optional path to save results CSV
        
    Returns:
        DataFrame with engineered features plus prediction scores
    """
    print(f"\n{'='*60}")
    print("STEP 1: FEATURE ENGINEERING")
    print(f"{'='*60}")
    print(f"Processing CSV: {csv_path}")
    
    # Run feature engineering pipeline
    try:
        features_df = engineer_features_from_csv(csv_path)
        print(f"\n✓ Feature engineering complete: {len(features_df)} companies, {len(features_df.columns)} features")
        
        # Save engineered features to CSV (preprocessing output)
        input_filename = os.path.splitext(os.path.basename(csv_path))[0]
        features_output_path = f"{input_filename}_company_features.csv"
        features_df.to_csv(features_output_path, index=False, encoding='utf-8-sig')
        print(f"✓ Preprocessing output saved to: {features_output_path}")
    except Exception as e:
        print(f"\n✗ Feature engineering failed: {e}")
        raise
    
    # Convert engineered features to agent format
    print(f"\n{'='*60}")
    print("STEP 2: PREDICTION")
    print(f"{'='*60}")
    datasets = convert_engineered_features_to_agent_format(features_df)
    print(f"Loaded {len(datasets)} companies to analyze\n")
    
    # Store results
    results = []
    
    for i, dataset in enumerate(datasets, 1):
        print(f"\n{'='*60}")
        print(f"Processing Company {i}/{len(datasets)}")
        print(f"{'='*60}")
        
        company_id = dataset['company_info'].get('company_id', 
                       dataset['company_info'].get('company_name', f'Company_{i}'))
        print(f"Company ID: {company_id}")
        
        try:
            # Run prediction
            result = agent.predict(
                target_department=target_department,
                product_description=product_description,
                raw_dataset=dataset
            )
            
            # Get the corresponding row from features_df to preserve all engineered features
            company_row = features_df.iloc[i-1].to_dict()
            
            # Extract scores
            purchase_score = result.get('purchase_likelihood_score', 'N/A')
            bant_score = result.get('bant_score', 'N/A')
            sentiment_score = result.get('sentiment_score', 'N/A')
            
            print(f"✓ Purchase Likelihood Score: {purchase_score}")
            print(f"  BANT Score: {bant_score}")
            print(f"  Sentiment Score: {sentiment_score}")
            
            # Add prediction scores
            result_row = {
                **company_row,
                'purchase_likelihood_score': purchase_score,
                'bant_score': bant_score,
                'sentiment_score': sentiment_score
            }
            
        except Exception as e:
            print(f"✗ Error: {str(e)}")
            # Get the corresponding row from features_df
            company_row = features_df.iloc[i-1].to_dict()
            
            result_row = {
                **company_row,
                'purchase_likelihood_score': 'ERROR',
                'bant_score': 'ERROR',
                'sentiment_score': 'ERROR',
                'error_message': str(e)
            }
        
        results.append(result_row)
    
    # Create results DataFrame
    results_df = pd.DataFrame(results)
    
    # Save to CSV if path provided
    if output_csv_path:
        results_df.to_csv(output_csv_path, index=False)
        print(f"\n✓ Final results saved to: {output_csv_path}")
    
    return results_df


def _create_agent_with_fallback() -> PurchaseLikelihoodAgent:
    """
    Helper to create the PurchaseLikelihoodAgent.
    
    Tries to use GPU mode first (if available), then falls back to CPU-only mode.
    This keeps setup simple for novice users.
    """
    try:
        print("\nAttempting to use GPU mode (if available)...")
        return PurchaseLikelihoodAgent(model_name="llama3.1:latest", num_gpu=1)
    except Exception as e:
        print(f"GPU mode failed: {e}")
        print("Falling back to CPU-only mode...")
        return PurchaseLikelihoodAgent(model_name="llama3.1:latest", num_gpu=0)


def _prompt_for_csv_path(max_attempts: int = 3) -> Optional[str]:
    """
    Prompt the user for a CSV file path and validate it.
    
    Returns a valid path string, or None if validation fails repeatedly.
    """
    for attempt in range(1, max_attempts + 1):
        print("\nPlease enter the path to your survey CSV file.")
        print("Example (Mac):  /Users/you/Documents/my_survey.csv")
        print("Example (relative):  data/survey.csv")
        csv_path_input = input("CSV file path: ").strip()

        if not csv_path_input:
            print("✗ You did not enter anything. Please type or paste a CSV file path.")
            continue

        # Expand ~ and convert to absolute path
        csv_path = os.path.abspath(os.path.expanduser(csv_path_input))

        # Basic validations
        if not csv_path.lower().endswith(".csv"):
            print(f"✗ The file must be a CSV (ending with .csv). You entered: {csv_path_input}")
            continue

        if not os.path.isfile(csv_path):
            print(f"✗ No file found at: {csv_path}")
            print("  Please check that the path is correct and the file exists.")
            continue

        print(f"✓ Found CSV file: {csv_path}")
        return csv_path

    print("\nToo many invalid attempts. Exiting without running predictions.")
    return None


def _prompt_for_product_description() -> str:
    """
    Prompt the user for a brief product description.
    """
    print("\nWhat is the product being sold? (e.g., 'AI-powered HR software for large enterprises')")
    description = input("Product Description: ").strip()
    while not description:
        print("Product description cannot be empty. Please provide a description.")
        description = input("Product Description: ").strip()
    print(f"✓ Using product description: {description}")
    return description


def _prompt_for_target_department(default: str = "IT") -> str:
    """
    Prompt the user for a target department, with a simple default.
    """
    print("\nWhich department is this product mainly for?")
    print(f"Press Enter to use the default: {default}")
    dept = input(f"Target department [{default}]: ").strip()
    if not dept:
        dept = default
    print(f"✓ Using target department: {dept}")
    return dept


def main() -> None:
    """
    Entry point for novice users.
    
    1. Shows brief setup info.
    2. Asks for a CSV file path and validates it.
    3. Asks for the target department (with default).
    4. Runs batch predictions and saves results to prediction_results.csv.
    """
    print("=" * 60)
    print("PURCHASE LIKELIHOOD PREDICTION AGENT")
    print("=" * 60)
    print("This tool uses an AI agent to analyze your survey CSV and")
    print("estimate how likely each company is to purchase your product.\n")

    print("SETUP (one-time):")
    print("1) Install Ollama from: https://ollama.com")
    print("2) In a terminal, run:")
    print("   - ollama pull llama3.2:8b")
    print("   - ollama pull phi3:mini")
    print("3) Start the Ollama server in a terminal:")
    print("   - ollama serve")
    print("If you see CUDA/GPU errors, don't worry – this script will")
    print("automatically fall back to CPU mode.\n")

    # Prompt user for CSV path
    csv_path = _prompt_for_csv_path()
    if csv_path is None:
        sys.exit(1)

    # Prompt for target department
    target_dept = _prompt_for_target_department(default="IT")

    # Prompt for product description
    product_desc = _prompt_for_product_description()

    # Create agent
    agent = _create_agent_with_fallback()

    print("\n" + "=" * 60)
    print("RUNNING BATCH PREDICTION")
    print("=" * 60)

    try:
        results_df = batch_predict_from_csv(
            agent=agent,
            csv_path=csv_path,
            target_department=target_dept,
            product_description=product_desc,
            output_csv_path="prediction_results.csv",
        )
    except Exception as e:
        print(f"\n✗ An unexpected error occurred while running predictions: {e}")
        sys.exit(1)

    print("\n" + "=" * 60)
    print("DONE")
    print("=" * 60)
    print("A summary of the first few rows:")
    try:
        preview_cols = [
            col
            for col in ["company_name", "company", "purchase_likelihood_score", "bant_score", "sentiment_score"]
            if col in results_df.columns
        ]
        if preview_cols:
            print(results_df[preview_cols].head().to_string(index=False))
        else:
            print(results_df.head().to_string(index=False))
    except Exception:
        pass

    print("\n✓ Full results have been saved to: prediction_results.csv")
    print("You can open this file in Excel, Numbers, or any spreadsheet tool.")


if __name__ == "__main__":
    main()