from langchain.tools import BaseTool
from pydantic import BaseModel, Field
from typing import Dict, Any, List, ClassVar
import pandas as pd
import json
import pickle

class PurchaseLikelihoodInput(BaseModel):
    """Input schema for the Purchase Likelihood ML Model."""
    company_info: Dict[str, Any] | str = Field(
        description="Company information as a dictionary or JSON string. Can include any fields like company_size, industry, company_name, revenue, etc. Keep it simple - just the key company facts."
    )
    bant_score: float = Field(
        description="BANT qualification score from 0 to 100"
    )
    sentiment_score: float = Field(
        description="Sentiment analysis score from 0 to 100"
    )


class PurchaseLikelihoodMLModel(BaseTool):
    """
    ML Model tool that predicts purchase likelihood based on company info,
    BANT score, and Sentiment score using a trained Random Forest model.
    """
    name: str = "purchase_likelihood_ml_model"
    description: str = """
    Predicts the likelihood (0-100) of a company purchasing a product using a trained ML model.
    Takes three separate inputs:
    1. company_info - A JSON string with company details (flexible structure)
    2. bant_score - BANT qualification score (0-100) 
    3. sentiment_score - Sentiment analysis score (0-100)
    
    Example usage:
    company_info='{"company_size": "Large", "industry": "Technology"}'
    bant_score=75.5
    sentiment_score=80.0
    """
    args_schema: type[BaseModel] = PurchaseLikelihoodInput
    
    # Class variable to store the loaded model
    _model: ClassVar[Any] = None
    _model_features: ClassVar[List[str]] = None
    
    # Translation mappings for Chinese inputs
    SIZE_TRANSLATION: ClassVar[Dict[str, str]] = {
        '大型企业': 'Enterprise',
        '大型': 'Enterprise',
        '企业级': 'Enterprise',
        '企业': 'Enterprise',
        '中型企业': 'Medium',
        '中型': 'Medium',
        '小型企业': 'Small',
        '小型': 'Small',
        '初创企业': 'Startup',
        '初创': 'Startup',
        '创业公司': 'Startup',
    }
    
    INDUSTRY_TRANSLATION: ClassVar[Dict[str, str]] = {
        '科技': 'Technology',
        '技术': 'Technology',
        '互联网': 'Technology',
        '金融': 'Finance',
        '医疗': 'Healthcare',
        '医疗保健': 'Healthcare',
        '制造': 'Manufacturing',
        '制造业': 'Manufacturing',
        '零售': 'Retail',
        '零售业': 'Retail',
        '教育': 'Education',
    }
    
    @classmethod
    def load_model(cls, model_path: str = 'backend/tools/intent_predictor/final_intent_model.pkl'):
        """Load the trained ML model from disk."""
        if cls._model is None:
            try:
                with open(model_path, 'rb') as f:
                    cls._model = pickle.load(f)
                print(f"✓ ML model loaded successfully from {model_path}")
                
                # Expected features based on your training code
                cls._model_features = [
                    'bant_score',
                    'sentiment_score',
                    'company_size',
                    'capital_millions',
                    'manager_ratio',
                    'director_ratio',
                    'has_email',
                    'total_calls',
                    'calls_per_day',
                    'response_rate',
                    'notes_ratio',
                    'max_questions_answered'
                ]
                print(f"✓ Model expects {len(cls._model_features)} features")
            except FileNotFoundError:
                raise FileNotFoundError(
                    f"Model file not found at {model_path}. "
                    "Please ensure the model is trained and saved at the correct location."
                )
            except Exception as e:
                raise RuntimeError(f"Error loading model: {str(e)}")
    
    def _translate_to_english(self, text: str, translation_map: Dict[str, str]) -> str:
        """Translate Chinese text to English using the provided mapping."""
        if not text:
            return 'Unknown'
        
        # Check if exact match exists
        if text in translation_map:
            return translation_map[text]
        
        # Check if any Chinese characters exist, if so look for partial matches
        if any('\u4e00' <= char <= '\u9fff' for char in text):
            for chinese, english in translation_map.items():
                if chinese in text:
                    return english
        
        # Return original if no translation found (might already be in English)
        return text
    
    def _parse_company_info(self, company_info_input: Dict[str, Any] | str) -> Dict[str, Any]:
        """Parse and normalize company info from dict or JSON string."""
        try:
            # Handle both dict and string inputs
            if isinstance(company_info_input, dict):
                company_info = company_info_input
            elif isinstance(company_info_input, str):
                company_info = json.loads(company_info_input)
            else:
                raise ValueError(f"company_info must be a dict or JSON string, got {type(company_info_input)}")
                
            # Normalize the parsed data
            normalized_info = {}
            
            for key, value in company_info.items():
                # Translate Chinese values to English for known fields
                if key in ['company_size', 'size', '公司规模', '规模']:
                    normalized_key = 'company_size'
                    normalized_value = self._translate_to_english(str(value), self.SIZE_TRANSLATION)
                    normalized_info[normalized_key] = value
                    normalized_info[f"{normalized_key}_normalized"] = normalized_value
                    
                elif key in ['industry', '行业', '产业']:
                    normalized_key = 'industry'
                    normalized_value = self._translate_to_english(str(value), self.INDUSTRY_TRANSLATION)
                    normalized_info[normalized_key] = value
                    normalized_info[f"{normalized_key}_normalized"] = normalized_value
                    
                else:
                    # Keep other fields as-is
                    normalized_info[key] = value
            
            return normalized_info
            
        except (json.JSONDecodeError, ValueError, AttributeError) as e:
            raise ValueError(f"Invalid company_info: {str(e)}")
    
    def _prepare_features(
        self,
        company_info: Dict[str, Any],
        bant_score: float,
        sentiment_score: float
    ) -> pd.DataFrame:
        """
        Prepare feature vector for the ML model.
        
        Args:
            company_info: Parsed company information
            bant_score: BANT score (0-100)
            sentiment_score: Sentiment score (0-100)
            
        Returns:
            DataFrame with features in the correct order for the model
        """
        # Initialize feature dictionary with defaults
        features = {
            'bant_score': bant_score,
            'sentiment_score': sentiment_score,
            'company_size': 0.0,
            'capital_millions': 0.0,
            'manager_ratio': 0.0,
            'director_ratio': 0.0,
            'has_email': 0.0,
            'total_calls': 0.0,
            'calls_per_day': 0.0,
            'response_rate': 0.0,
            'notes_ratio': 0.0,
            'max_questions_answered': 0.0
        }
        
        # Extract available features from company_info
        # Map company_size to numeric value
        if 'company_size' in company_info:
            size_mapping = {
                'Enterprise': 5.0,
                'Large': 4.0,
                'Medium': 3.0,
                'Small': 2.0,
                'Startup': 1.0
            }
            size_str = str(company_info.get('company_size', '')).strip()
            features['company_size'] = size_mapping.get(size_str, 3.0)  # Default to Medium
        
        # Map other available fields
        field_mappings = {
            'capital_millions': 'capital_millions',
            'manager_ratio': 'manager_ratio',
            'director_ratio': 'director_ratio',
            'has_email': 'has_email',
            'total_calls': 'total_calls',
            'calls_per_day': 'calls_per_day',
            'response_rate': 'response_rate',
            'notes_ratio': 'notes_ratio',
            'max_questions_answered': 'max_questions_answered'
        }
        
        for company_key, feature_key in field_mappings.items():
            if company_key in company_info:
                try:
                    features[feature_key] = float(company_info[company_key])
                except (ValueError, TypeError):
                    pass  # Keep default value
        
        # Create DataFrame with features in the correct order
        feature_df = pd.DataFrame([features], columns=self._model_features)
        
        return feature_df
    
    def _run(
        self,
        company_info: Dict[str, Any] | str,
        bant_score: float,
        sentiment_score: float
    ) -> str:
        """
        Execute the ML model prediction with structured inputs.
        
        Args:
            company_info: Dictionary or JSON string with flexible company fields
            bant_score: BANT qualification score (0-100)
            sentiment_score: Sentiment analysis score (0-100)
        """
        try:
            # Load model if not already loaded
            self.load_model()
            
            # Parse and normalize company info
            parsed_company_info = self._parse_company_info(company_info)
            
            # Validate and normalize scores
            bant_score = max(0, min(100, float(bant_score)))
            sentiment_score = max(0, min(100, float(sentiment_score)))
            
            # Prepare features for ML model
            feature_df = self._prepare_features(parsed_company_info, bant_score, sentiment_score)
            
            # Get prediction from the trained model
            # predict_proba returns probabilities for each class [prob_class_0, prob_class_1]
            probabilities = self._model.predict_proba(feature_df)
            purchase_probability = probabilities[0, 1]  # Probability of class 1 (purchase)
            
            # Convert to 0-100 scale
            score = round(purchase_probability * 100, 2)
            
            # Build result
            result = {
                "purchase_likelihood_score": score,
                "bant_score": round(bant_score, 2),
                "sentiment_score": round(sentiment_score, 2),
                "company_info": parsed_company_info,
                "model_info": {
                    "model_type": "Random Forest Classifier",
                    "prediction_confidence": round(max(probabilities[0]) * 100, 2)
                }
            }
            
            return json.dumps(result, ensure_ascii=False, indent=2)
            
        except (ValueError, TypeError, json.JSONDecodeError) as e:
            return json.dumps({
                "error": f"Invalid input: {str(e)}",
                "received_inputs": {
                    "company_info": company_info,
                    "bant_score": bant_score,
                    "sentiment_score": sentiment_score
                },
                "hint": "Provide three separate parameters: company_info (JSON string), bant_score (number), sentiment_score (number)"
            }, ensure_ascii=False, indent=2)
        except Exception as e:
            return json.dumps({
                "error": f"Model prediction failed: {str(e)}",
                "hint": "Ensure the ML model is properly trained and saved at backend/tools/intent_predictor/final_intent_model.pkl"
            }, ensure_ascii=False, indent=2)
