"""
Universal Feature Engineering for Survey Data
==============================================

This module provides feature engineering functions for survey data processing.
It can be used as a library or run standalone.

Main function: engineer_features_from_csv(csv_path) - processes a single CSV file
and returns engineered company-level features.

Features:
- Uses Gemini 2.5 Flash LLM to auto-convert different column formats to standard format
- Engineers 45+ features per company
- Distinguishes between "question exists but no answer" vs "question not in survey"

Author: Data Analysis Team
Date: November 2025
"""

import os
import json
from typing import Any
import pandas as pd
import numpy as np
import warnings
import requests
import re
from dotenv import load_dotenv

warnings.filterwarnings('ignore')

# Load API key from .env file
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_API_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent"

# Special marker for question responses
NO_ANSWER = "[NO_ANSWER]"      # Question exists but no response


def call_gemini_api(prompt, max_tokens=4096):
    """
    Call Gemini 2.5 Flash API to get response.
    
    Args:
        prompt: The prompt to send to the API
        max_tokens: Maximum tokens in response
        
    Returns:
        API response text or None if failed
    """
    if not GEMINI_API_KEY:
        print("  ⚠️ GEMINI_API_KEY not found in .env file")
        return None
    
    headers = {
        "Content-Type": "application/json"
    }
    
    data = {
        "contents": [
            {
                "parts": [
                    {"text": prompt}
                ]
            }
        ],
        "generationConfig": {
            "temperature": 0.1,
            "maxOutputTokens": max_tokens
        }
    }
    
    try:
        response = requests.post(
            f"{GEMINI_API_URL}?key={GEMINI_API_KEY}",
            headers=headers,
            json=data,
            timeout=60
        )
        
        if response.status_code == 200:
            result = response.json()
            if 'candidates' in result and len(result['candidates']) > 0:
                return result['candidates'][0]['content']['parts'][0]['text']
        else:
            print(f"  ⚠️ API Error: {response.status_code} - {response.text[:200]}")
            return None
    except Exception as e:
        print(f"  ⚠️ API Exception: {str(e)}")
        return None


def generate_column_mapping_code(columns_json, expected_columns):
    """
    Use LLM to generate Python code for mapping columns to expected format.
    
    Args:
        columns_json: JSON string of actual column names
        expected_columns: JSON string of expected column names
        
    Returns:
        Python code string for column mapping, or None
    """
    prompt = f"""You are a Python programmer. I need to map CSV columns from one format to another.

    ACTUAL columns in the file:
    {columns_json}

    EXPECTED columns for processing:
    {expected_columns}

    Generate a Python dictionary called `column_mapping` that maps ACTUAL column names to EXPECTED column names.
    Only include mappings where there's a clear match. If a column cannot be mapped, don't include it.

    Common mappings to look for:
    - "Company ID" or similar -> "Company ID"
    - "Call Date" or "call_date" -> "Call Date"  
    - "Call Time" or "call_time" -> "Call Time"
    - "名單狀態" or similar status -> "名單狀態"
    - "Call狀態" or similar -> "Call狀態"
    - "聯絡人狀態" or similar -> "聯絡人狀態"
    - "員工人數" or similar -> "員工人數"
    - "資本額" or similar -> "資本額"
    - "部門" or similar -> "部門"
    - "職稱" or similar -> "職稱"
    - "Email" or "email" or "E-mail" -> "Email"
    - "SIC" or similar -> "SIC"
    - "營業項目" or similar -> "營業項目"

    Return ONLY valid Python code that defines the column_mapping dictionary. Example:
    ```python
    column_mapping = {{
        "Company ID": "Company ID",
        "Call Date": "Call Date",
        "員工人數": "員工人數"
    }}
    ```

    If most columns already match, just return an empty mapping:
    ```python
    column_mapping = {{}}
    ```

    Python code:"""

    response = call_gemini_api(prompt)
    
    if response:
        # Extract Python code from response
        try:
            # Find code block
            code_match = re.search(r'```python\s*(.*?)\s*```', response, re.DOTALL)
            if code_match:
                return code_match.group(1)
            # Try without code block markers
            dict_match = re.search(r'column_mapping\s*=\s*\{[^}]*\}', response, re.DOTALL)
            if dict_match:
                return dict_match.group()
        except Exception:
            pass
    
    return "column_mapping = {}"


def simplify_question_name(question_col):
    """
    Simplify a question column name for output.
    Keeps full question name without truncation.
    
    Args:
        question_col: Original question column name
        
    Returns:
        Simplified column name (full length preserved)
    """
    # Remove common prefixes
    simplified = question_col
    
    # Remove "Q." or "Q" prefix with number
    simplified = re.sub(r'^Q\.?\s*\d*[\.\-]?\s*', '', simplified)
    
    # Remove question mark
    simplified = simplified.replace('？', '').replace('?', '')
    
    # Clean up whitespace (no truncation - keep full name)
    simplified = simplified.strip()
    
    return simplified if simplified else question_col


def infer_question_columns(df: pd.DataFrame):
    """
    Infer survey question columns directly from a single DataFrame.
    
    Heuristics:
    - Column name starts with 'Q' or 'Q.' (common survey question style)
    - OR column name contains '?' or '？'
    - Excludes obvious note/remark columns.
    """
    question_columns = []
    for col in df.columns:
        col_str = str(col)
        lower = col_str.lower()
        is_note = ('備註' in col_str) or ('note' in lower) or ('remark' in lower)
        if is_note:
            continue
        if col_str.startswith('Q') or '？' in col_str or '?' in col_str:
            question_columns.append(col)
    return question_columns


def load_and_standardize_data(input_data):
    """
    Load a CSV file and standardize column names using LLM if needed.
    
    Args:
        input_data: a pandas DataFrame from the frontend.
        
    Returns:
        Tuple of (Standardized DataFrame, original_columns, question_columns_in_file)
    """
    # Load csv to df
    print("\n  Processing uploaded DataFrame from UI...")
    df = input_data.copy()

    original_columns = list(df.columns)
    
    # Expected column names (based on original ERP format)
    expected_columns = [
        "Company ID", "Call Date", "Call Time", "名單狀態", "聯絡人狀態", 
        "Call狀態", "員工人數", "資本額", "部門", "職稱", "Email", 
        "SIC", "營業項目"
    ]
    
    # Check if column mapping is needed
    missing_expected = [col for col in expected_columns if col not in original_columns]
    
    if missing_expected:
        print(f"    → Some expected columns missing, attempting auto-mapping...")
        
        # Get column mapping from LLM
        columns_json = json.dumps(original_columns, ensure_ascii=False)
        expected_json = json.dumps(expected_columns, ensure_ascii=False)
        
        mapping_code = generate_column_mapping_code(columns_json, expected_json)
        
        try:
            # Execute the mapping code
            local_vars = {}
            exec(mapping_code, {}, local_vars)
            column_mapping = local_vars.get('column_mapping', {})
            
            if column_mapping:
                print(f"    → Mapping {len(column_mapping)} columns")
                # Rename columns
                df = df.rename(columns=column_mapping)
        except Exception as e:
            print(f"    ⚠️ Mapping failed: {e}")
    
    return df, original_columns







def preprocess_data(df, survey_name):
    """
    Preprocess a DataFrame with standardized feature engineering.
    
    Args:
        df: Raw DataFrame
        survey_name: Name of the survey
        question_mapping: Dict of {simplified_name: original_col}
        questions_in_file: Set of question names that exist in this file
        
    Returns:
        Preprocessed DataFrame
    """
    df = df.copy()
    
    # Add survey identifier
    df['Survey_Name'] = survey_name
    
    # Parse datetime
    if 'Call Date' in df.columns:
        time_col = 'Call Time' if 'Call Time' in df.columns else 'call_time'
        if time_col in df.columns:
            df['call_datetime'] = pd.to_datetime(
                df['Call Date'].astype(str) + ' ' + df[time_col].astype(str),
                errors='coerce'
            )
        else:
            df['call_datetime'] = pd.to_datetime(df['Call Date'], errors='coerce')
    else:
        df['call_datetime'] = pd.NaT
    
    # Standardize company ID
    if 'Company ID' in df.columns:
        df['company_id'] = df['Company ID']
    
    # Identify completion status
    completion_columns = ['名單狀態', 'Call狀態', '聯絡人狀態']
    df['is_completed'] = False
    for col in completion_columns:
        if col in df.columns:
            df['is_completed'] = df['is_completed'] | df[col].str.contains('完成問卷', na=False)
    
    # Extract company size
    if '員工人數' in df.columns:
        df['company_size'] = pd.to_numeric(df['員工人數'], errors='coerce')
    else:
        df['company_size'] = np.nan
    
    # Extract capital
    if '資本額' in df.columns:
        df['capital'] = pd.to_numeric(df['資本額'], errors='coerce')
    else:
        df['capital'] = np.nan
    
    # Extract industry info
    df['sic_code'] = df['SIC'] if 'SIC' in df.columns else np.nan
    df['industry'] = df['營業項目'] if '營業項目' in df.columns else np.nan
    
    # Contact department
    if '部門' in df.columns:
        df['is_it_department'] = df['部門'].str.contains('資訊|資料|IT|MIS', na=False, case=False)
    else:
        df['is_it_department'] = False
    
    # Contact title level
    if '職稱' in df.columns:
        title = df['職稱'].fillna('')
        df['is_manager'] = title.str.contains('經理|總經理|副總|協理|長|主管|主任', na=False)
        df['is_director'] = title.str.contains('長|總經理|副總|協理', na=False)
    else:
        df['is_manager'] = False
        df['is_director'] = False
    
    # Call status patterns
    if 'Call狀態' in df.columns:
        df['is_cannot_contact'] = df['Call狀態'].str.contains('無法聯繫', na=False)
        df['is_refused'] = df['Call狀態'].str.contains('拒絕', na=False)
        df['is_wrong_number'] = df['Call狀態'].str.contains('錯號', na=False)
    else:
        df['is_cannot_contact'] = False
        df['is_refused'] = False
        df['is_wrong_number'] = False
    
    # Email analysis
    email_col = None
    for col in ['Email', 'email', 'E-mail']:
        if col in df.columns:
            email_col = col
            break
    
    if email_col:
        df['has_email'] = df[email_col].notna() & (df[email_col].astype(str).str.len() > 3)
        
        def extract_email_domain(email):
            if pd.isna(email) or str(email).strip() == '':
                return 'none'
            email_str = str(email).lower().strip()
            if '@' not in email_str:
                return 'invalid'
            domain = email_str.split('@')[-1].split('.')[0]
            return domain
        
        df['email_domain'] = df[email_col].apply(extract_email_domain)
        
        def categorize_email_type(domain):
            if domain in ['none', 'invalid']:
                return 'none'
            free_providers = ['gmail', 'yahoo', 'hotmail', 'outlook', 'icloud', 'me', 'live', 'msn', 'hinet']
            if domain in free_providers:
                return 'personal'
            return 'work'
        
        df['email_type'] = df['email_domain'].apply(categorize_email_type)
        df['has_work_email'] = df['email_type'] == 'work'
        df['has_personal_email'] = df['email_type'] == 'personal'
    else:
        df['has_email'] = False
        df['email_domain'] = 'none'
        df['email_type'] = 'none'
        df['has_work_email'] = False
        df['has_personal_email'] = False
    
    # Survey response depth
    question_columns = infer_question_columns(df)
    df['questions_answered'] = 0
    if question_columns:
        for col in question_columns:
            df['questions_answered'] += df[col].notna() & (df[col].astype(str).str.strip() != '')
    
    df['has_survey_responses'] = df['questions_answered'] > 0
    
    # Notes/remarks
    note_columns = [col for col in df.columns if '備註' in col or 'note' in col.lower() or 'remark' in col.lower()]
    df['has_notes'] = False
    if note_columns:
        for col in note_columns:
            df['has_notes'] = df['has_notes'] | (df[col].notna() & (df[col].astype(str).str.len() > 5))
    
    return df


def engineer_call_features(df):
    """
    Engineer call-level features.
    """
    df = df.sort_values(['company_id', 'call_datetime']).reset_index(drop=True)
    
    # Call sequence
    df['call_sequence'] = df.groupby('company_id').cumcount() + 1
    
    # Time intervals
    df['prev_call_datetime'] = df.groupby('company_id')['call_datetime'].shift(1)
    df['days_since_prev_call'] = (df['call_datetime'] - df['prev_call_datetime']).dt.total_seconds() / 86400
    
    # Call timing
    df['call_hour'] = df['call_datetime'].dt.hour
    df['is_morning'] = (df['call_hour'] >= 9) & (df['call_hour'] < 12)
    df['is_afternoon'] = (df['call_hour'] >= 14) & (df['call_hour'] < 17)
    df['call_weekday'] = df['call_datetime'].dt.dayofweek
    
    return df


def create_company_features(df):
    """
    Aggregate call-level data to create company-level features.
    """
    print("\n  Creating company-level features...")

    # Determine question columns directly from the combined DataFrame
    question_columns = infer_question_columns(df)
    
    company_features = []
    
    for company_id, group in df.groupby('company_id'):
        features = {'company_id': company_id}
        
        # Basic info
        features['survey_name'] = group['Survey_Name'].iloc[0]
        features['total_calls'] = len(group)
        features['completed_survey'] = group['is_completed'].any()
        
        # Call pattern features
        features['call_sequence_max'] = group['call_sequence'].max()
        
        if len(group) > 1:
            features['days_between_calls_mean'] = group['days_since_prev_call'].mean()
            features['days_between_calls_std'] = group['days_since_prev_call'].std()
            features['days_between_calls_max'] = group['days_since_prev_call'].max()
        else:
            features['days_between_calls_mean'] = 0
            features['days_between_calls_std'] = 0
            features['days_between_calls_max'] = 0
        
        date_range = (group['call_datetime'].max() - group['call_datetime'].min()).total_seconds() / 86400
        if date_range > 0:
            features['calls_per_day'] = len(group) / date_range
            features['campaign_duration_days'] = date_range
        else:
            features['calls_per_day'] = len(group)
            features['campaign_duration_days'] = 1
        
        features['morning_calls_ratio'] = group['is_morning'].mean()
        features['afternoon_calls_ratio'] = group['is_afternoon'].mean()
        features['avg_call_hour'] = group['call_hour'].mean()
        features['weekday_calls_ratio'] = (group['call_weekday'] < 5).mean()
        
        features['cannot_contact_ratio'] = group['is_cannot_contact'].mean()
        features['refused_ratio'] = group['is_refused'].mean()
        features['wrong_number_ratio'] = group['is_wrong_number'].mean()
        features['response_rate'] = 1 - features['cannot_contact_ratio']
        
        first_2_calls = group.head(2)
        features['early_cannot_contact'] = first_2_calls['is_cannot_contact'].mean()
        features['early_refused'] = first_2_calls['is_refused'].mean()
        features['early_response_rate'] = 1 - features['early_cannot_contact']
        features['completed_in_first_2_calls'] = first_2_calls['is_completed'].any()
        
        # Company characteristics
        features['company_size'] = group['company_size'].iloc[0] if pd.notna(group['company_size'].iloc[0]) else 0
        features['company_size_category'] = 'unknown'
        if features['company_size'] >= 1000:
            features['company_size_category'] = 'large'
        elif features['company_size'] >= 200:
            features['company_size_category'] = 'medium'
        elif features['company_size'] > 0:
            features['company_size_category'] = 'small'
        
        features['capital_millions'] = group['capital'].iloc[0] / 1_000_000 if pd.notna(group['capital'].iloc[0]) else 0
        features['sic_code'] = group['sic_code'].iloc[0] if pd.notna(group['sic_code'].iloc[0]) else 'unknown'
        features['industry'] = group['industry'].iloc[0] if pd.notna(group['industry'].iloc[0]) else 'unknown'
        
        # Contact quality
        features['contacted_it_dept'] = group['is_it_department'].any()
        features['it_dept_ratio'] = group['is_it_department'].mean()
        features['contacted_manager'] = group['is_manager'].any()
        features['manager_ratio'] = group['is_manager'].mean()
        features['contacted_director'] = group['is_director'].any()
        features['director_ratio'] = group['is_director'].mean()
        features['persistent_campaign'] = (len(group) >= 3)
        
        # Email features
        features['has_email'] = group['has_email'].any()
        features['email_provided_ratio'] = group['has_email'].mean()
        features['has_work_email'] = group['has_work_email'].any()
        features['has_personal_email'] = group['has_personal_email'].any()
        
        email_domains = group[group['email_domain'] != 'none']['email_domain']
        features['primary_email_domain'] = email_domains.mode().iloc[0] if len(email_domains) > 0 else 'none'
        
        # Survey response depth
        features['max_questions_answered'] = group['questions_answered'].max()
        features['avg_questions_answered'] = group['questions_answered'].mean()
        features['has_survey_responses'] = group['has_survey_responses'].any()
        
        if features['completed_survey']:
            completed_calls = group[group['is_completed'] == True]
            features['completion_questions_count'] = completed_calls['questions_answered'].max() if len(completed_calls) > 0 else 0
        else:
            features['completion_questions_count'] = 0
        
        # Notes features
        features['has_notes'] = group['has_notes'].any()
        features['notes_ratio'] = group['has_notes'].mean()

        # Aggregate survey question responses for this company.
        # For each question column, take the latest non-empty answer;
        # if the question exists but has no answers, tag as NO_ANSWER.
        if question_columns:
            # Ensure newest calls first
            group_sorted = group.sort_values('call_datetime', ascending=False)
            for q_col in question_columns:
                simplified_name = simplify_question_name(q_col)
                feature_col = f"Q_{simplified_name}"

                value_set = False
                for _, row in group_sorted.iterrows():
                    val = row.get(q_col)
                    if pd.notna(val) and str(val).strip():
                        features[feature_col] = str(val).strip()
                        value_set = True
                        break

                if not value_set:
                    # Question column exists for this company but no answer recorded
                    features[feature_col] = NO_ANSWER
        
        company_features.append(features)
    
    result_df = pd.DataFrame(company_features)
    print(f"  → Engineered features for {len(result_df):,} companies ({len(result_df.columns)} columns)")
    
    return result_df


def engineer_features_from_csv(input_data) -> pd.DataFrame:
    """
    Process a single CSV file through the full feature engineering pipeline.
    
    This is the main entry point for feature engineering from a CSV file.
    It runs: load_and_standardize_data → preprocess_data → engineer_call_features → create_company_features
    
    Args:
        input_data: the pandas DataFrame from UI
        
    Returns:
        DataFrame with engineered company-level features
    """
    # Handle naming and display logic
    if isinstance(input_data, pd.DataFrame):
        display_name = "UI_Upload"
        survey_name = "UI_Survey"
    else:
        display_name = os.path.basename(input_data)
        # Get survey name from filename logic
        survey_name = display_name.split(' - ')[-1].replace('.csv', '') if ' - ' in display_name else display_name.replace('.csv', '')
    
    survey_name = survey_name[:50]  # Truncate for safety

    print(f"\n{'='*80}")
    print(f"FEATURE ENGINEERING: {display_name}")
    print(f"{'='*80}")

    # Load and standardize (using the updated function from the previous step)
    df, original_columns = load_and_standardize_data(input_data)
    
    # Preprocess and engineer call-level features
    processed = preprocess_data(df, survey_name)
    processed = engineer_call_features(processed)
    print(f"    → Processed {len(processed):,} call records")
    
    # Create company features directly from this single file
    company_df = create_company_features(processed)
    
    return company_df
