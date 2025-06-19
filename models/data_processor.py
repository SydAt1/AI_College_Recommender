import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
import ast
import re
from scipy.sparse import csr_matrix

class CollegeDataProcessor:
    """Data processor for college recommendation system"""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.tfidf_vectorizer = TfidfVectorizer(max_features=50, stop_words='english')
        self.tfidf_fitted = False
        self.scaler_fitted = False
        
    def load_data(self, filepath=None):
        """Load college data from CSV file"""
        if filepath is None:
            # Try multiple possible paths for the data file
            possible_paths = [
                os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'colleges.csv'),
                os.path.join(os.path.dirname(__file__), '..', 'data', 'colleges.csv'),
                'data/colleges.csv',
                '../data/colleges.csv'
            ]
            
            for path in possible_paths:
                if os.path.exists(path):
                    filepath = path
                    break
            else:
                # If no path found, use the original logic
                filepath = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'colleges.csv')
        
        try:
            df = pd.read_csv(filepath)
            print(f"Loaded {len(df)} college records from {filepath}")
            return df
        except FileNotFoundError:
            print(f"File {filepath} not found. Please run data/sample_data.py first.")
            return None
    
    def preprocess_data(self, df):
        """Preprocess the college data"""
        df_processed = df.copy()
        
        # Handle missing values
        numeric_columns = df_processed.select_dtypes(include=[np.number]).columns
        df_processed[numeric_columns] = df_processed[numeric_columns].fillna(df_processed[numeric_columns].median())
        
        # Convert top_majors from string to list if needed
        if 'top_majors' in df_processed.columns:
            df_processed['top_majors'] = df_processed['top_majors'].apply(
                lambda x: ast.literal_eval(x) if isinstance(x, str) else x
            )
        
        # Create derived features
        df_processed = self._create_derived_features(df_processed)
        
        return df_processed
    
    def _create_derived_features(self, df):
        """Create additional features for better recommendations"""
        
        # Cost categories
        df['cost_category'] = pd.cut(
            df['total_cost_in_state'], 
            bins=[0, 20000, 40000, 60000, 100000], 
            labels=['Low', 'Medium', 'High', 'Very High']
        )
        
        # Selectivity categories
        df['selectivity_category'] = pd.cut(
            df['selectivity_score'], 
            bins=[0, 0.3, 0.6, 0.8, 1.0], 
            labels=['Less Selective', 'Selective', 'Very Selective', 'Most Selective']
        )
        
        # Size categories
        df['size_category'] = pd.cut(
            df['student_population'], 
            bins=[0, 5000, 15000, 30000, 100000], 
            labels=['Small', 'Medium', 'Large', 'Very Large']
        )
        
        # Region (simplified)
        region_mapping = {
            'CA': 'West', 'OR': 'West', 'WA': 'West', 'AZ': 'West', 'CO': 'West', 'UT': 'West',
            'TX': 'South', 'FL': 'South', 'GA': 'South', 'NC': 'South', 'SC': 'South', 'VA': 'South',
            'NY': 'Northeast', 'MA': 'Northeast', 'CT': 'Northeast', 'NJ': 'Northeast', 'PA': 'Northeast',
            'IL': 'Midwest', 'MI': 'Midwest', 'OH': 'Midwest', 'IN': 'Midwest', 'WI': 'Midwest', 'MN': 'Midwest', 'IA': 'Midwest'
        }
        df['region'] = df['state'].map(region_mapping)
        
        # Academic strength score
        df['academic_strength'] = (
            df['selectivity_score'] * 0.4 + 
            df['graduation_rate'] * 0.3 + 
            df['retention_rate'] * 0.3
        )
        
        # Value score (quality vs cost)
        df['value_score'] = df['academic_strength'] / (df['total_cost_in_state'] / 10000)
        
        return df
    
    def encode_categorical_features(self, df):
        """Encode categorical features"""
        categorical_columns = [
            'state', 'campus_size', 'location_type', 'public_private', 
            'religious_affiliation', 'athletics_division', 'cost_category',
            'selectivity_category', 'size_category', 'region'
        ]
        
        df_encoded = df.copy()
        
        for col in categorical_columns:
            if col in df_encoded.columns:
                if col not in self.label_encoders:
                    self.label_encoders[col] = LabelEncoder()
                    df_encoded[col] = self.label_encoders[col].fit_transform(df_encoded[col].astype(str))
                else:
                    # Handle unseen categories
                    df_encoded[col] = df_encoded[col].astype(str)
                    df_encoded[col] = df_encoded[col].map(
                        lambda x: x if x in self.label_encoders[col].classes_ else 'Unknown'
                    )
                    df_encoded[col] = self.label_encoders[col].transform(df_encoded[col])
        
        return df_encoded
    
    def process_majors(self, df):
        """Process major information using TF-IDF"""
        if 'top_majors' in df.columns:
            # Convert list of majors to string
            majors_text = df['top_majors'].apply(lambda x: ' '.join(x) if isinstance(x, list) else str(x))
            
            if not self.tfidf_fitted:
                majors_tfidf = self.tfidf_vectorizer.fit_transform(majors_text)  # type: ignore
                self.tfidf_fitted = True
            else:
                majors_tfidf = self.tfidf_vectorizer.transform(majors_text)  # type: ignore
            
            # Convert to DataFrame
            majors_df = pd.DataFrame(
                majors_tfidf.toarray(),  # type: ignore
                columns=[f'major_{i}' for i in range(majors_tfidf.shape[1])],  # type: ignore
                index=df.index
            )
            
            return majors_df
        
        return pd.DataFrame()
    
    def scale_numeric_features(self, df):
        """Scale numeric features"""
        numeric_features = [
            'acceptance_rate', 'avg_sat', 'avg_act', 'avg_gpa',
            'tuition_in_state', 'tuition_out_state', 'room_board',
            'total_cost_in_state', 'total_cost_out_state', 'student_population',
            'graduation_rate', 'retention_rate', 'selectivity_score',
            'affordability_score', 'quality_score', 'academic_strength', 'value_score'
        ]
        
        available_features = [col for col in numeric_features if col in df.columns]
        
        if available_features:
            if not self.scaler_fitted:
                df[available_features] = self.scaler.fit_transform(df[available_features])
                self.scaler_fitted = True
            else:
                df[available_features] = self.scaler.transform(df[available_features])
        
        return df
    
    def prepare_features(self, df):
        """Prepare all features for the ML model"""
        # Preprocess data
        df_processed = self.preprocess_data(df)
        
        # Encode categorical features
        df_encoded = self.encode_categorical_features(df_processed)
        
        # Process majors
        majors_df = self.process_majors(df_processed)
        
        # Scale numeric features
        df_scaled = self.scale_numeric_features(df_encoded)
        
        # Combine all features
        feature_columns = [
            'acceptance_rate', 'avg_sat', 'avg_act', 'avg_gpa',
            'tuition_in_state', 'tuition_out_state', 'room_board',
            'total_cost_in_state', 'total_cost_out_state', 'student_population',
            'graduation_rate', 'retention_rate', 'selectivity_score',
            'affordability_score', 'quality_score', 'academic_strength', 'value_score',
            'state', 'campus_size', 'location_type', 'public_private',
            'religious_affiliation', 'athletics_division', 'cost_category',
            'selectivity_category', 'size_category', 'region'
        ]
        
        available_features = [col for col in feature_columns if col in df_scaled.columns]
        features_df = df_scaled[available_features].copy()
        
        # Add majors features
        if not majors_df.empty:
            features_df = pd.concat([features_df, majors_df], axis=1)
        
        return features_df, df_processed
    
    def get_feature_names(self):
        """Get list of feature names"""
        return self.scaler.get_feature_names_out() if hasattr(self.scaler, 'get_feature_names_out') else [] 