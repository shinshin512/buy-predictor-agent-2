import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# ==========================================
# 1. Load Data & Clean Survey Details
# ==========================================
input_file = 'prediction_results.csv'
print(f"🚀 Loading dataset: {input_file}...")

# Use low_memory=False to prevent mixed type warnings
df = pd.read_csv(input_file, low_memory=False)

print(f"   Original column count: {len(df.columns)}")

# --- Data Cleaning Logic: Remove detailed survey answers (Columns starting with Q_) ---
# We only keep columns that do NOT start with 'Q_'
cols_to_keep = [c for c in df.columns if not c.startswith('Q_')]
df_clean = df[cols_to_keep].copy()

print(f"   Cleaned column count: {len(df_clean.columns)} (Removed detailed Q_ columns)")

# ==========================================
# 2. Feature Selection
# ==========================================

# Define feature candidates including BANT and Sentiment scores
feature_candidates = [
    # --- Advanced Scores (Retrieved from dataset) ---
    'bant_score',            # Included
    'sentiment_score',       # Included
    
    # --- Firmographic (Static) ---
    'company_size',          
    'capital_millions',      
    
    # --- Contact Quality ---
    'manager_ratio',         
    'director_ratio',        
    'has_email',             
    
    # --- Engagement / Behavioral ---
    'total_calls',           
    'calls_per_day',         
    'response_rate',         
    'notes_ratio',  
    'max_questions_answered' 
]

# STRICTLY exclude 'purchase_likelihood_score' to prevent data leakage
blacklist = ['purchase_likelihood_score', 'Unnamed: 0']

# Final filtering of features
final_features = [f for f in feature_candidates if f in df_clean.columns and f not in blacklist]

print(f"✅ Final features selected for training ({len(final_features)}):")
print(final_features)

# Prepare X (Features) and y (Target)
# Convert to numeric and handle missing values
X = df_clean[final_features].apply(pd.to_numeric, errors='coerce').fillna(0)

# Verify Target Variable
target_col = 'completed_survey'
if target_col not in df_clean.columns:
    print(f"❌ ERROR: Target column '{target_col}' not found. Please check CSV.")
    exit()

# Convert target to integer (0/1)
y = df_clean[target_col].astype(int)

# ==========================================
# 3. Train Model
# ==========================================
print("🤖 Training AI Model...")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(
    n_estimators=200,       
    min_samples_leaf=10,    # Prevents overfitting; ensures BANT scores have impact
    max_depth=15,           
    random_state=42,
    n_jobs=-1               
)

model.fit(X_train, y_train)

# Validation
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"🏆 Model Accuracy: {acc:.2%}")

# ==========================================
# 4. Save Model & Generate Scores
# ==========================================
model_filename = 'final_intent_model.pkl'
with open(model_filename, 'wb') as f:
    pickle.dump(model, f)
print(f"💾 Final model saved as: {model_filename}")

print("⚡ Calculating Intent Scores...")
# Calculate probability of class 1 (Survey Completion)
probabilities = model.predict_proba(X)[:, 1]
df_clean['intent_score'] = (probabilities * 100).round(1)

# ==========================================
# 5. Feature Importance Analysis
# ==========================================
print("\n📊 Feature Importance Ranking (Top 5):")
importances = pd.DataFrame({
    'feature': final_features,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)
print(importances.head(5))

# ==========================================
# 6. Export Results
# ==========================================
output_file = 'scored_leads_final.csv'
cols_export = ['company_id', 'intent_score', 'completed_survey'] + final_features

df_clean.sort_values(by='intent_score', ascending=False)[cols_export].to_csv(output_file, index=False)

print(f"✅ Process Complete! Final scored list saved to: {output_file}")