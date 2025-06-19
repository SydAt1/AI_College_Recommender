import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import joblib
import os
from typing import List, Dict, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

class CollegeRecommendationModel:
    """ML model for college recommendations"""
    
    def __init__(self, model_type='hybrid'):
        self.model_type = model_type
        self.models = {}
        self.scaler = StandardScaler()
        self.is_trained = False
        self.feature_names = []
        self.college_data = None
        self.raw_college_data = None
        
    def train(self, features_df: pd.DataFrame, college_data: pd.DataFrame, 
              target_column: str = 'quality_score', raw_college_data: pd.DataFrame = None):
        """Train the recommendation model"""
        # Store the processed college data (with encoded features) for ML
        self.college_data = college_data.copy()
        # Store raw college data for returning college information
        self.raw_college_data = raw_college_data.copy() if raw_college_data is not None else college_data.copy()
        self.feature_names = features_df.columns.tolist()
        
        # Prepare target variable
        if target_column in college_data.columns:
            target = college_data[target_column]
        else:
            # Create synthetic target based on multiple factors
            target = self._create_synthetic_target(college_data)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            features_df, target, test_size=0.2, random_state=42
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train different models based on type
        if self.model_type == 'hybrid':
            self._train_hybrid_model(X_train_scaled, y_train, X_test_scaled, y_test)
        elif self.model_type == 'collaborative':
            self._train_collaborative_model(features_df, target)
        elif self.model_type == 'content_based':
            self._train_content_based_model(features_df, target)
        else:
            self._train_single_model(X_train_scaled, y_train, X_test_scaled, y_test)
        
        self.is_trained = True
        print(f"Model training completed. Type: {self.model_type}")
        
    def _create_synthetic_target(self, college_data: pd.DataFrame) -> pd.Series:
        """Create a synthetic target variable for training"""
        # Combine multiple factors to create a comprehensive score
        target = (
            college_data['quality_score'] * 0.4 +
            college_data['affordability_score'] * 0.3 +
            college_data['selectivity_score'] * 0.2 +
            college_data['graduation_rate'] * 0.1
        )
        return target
    
    def _train_hybrid_model(self, X_train, y_train, X_test, y_test):
        """Train a hybrid model combining multiple approaches"""
        
        # 1. Random Forest for feature importance
        rf_model = RandomForestRegressor(
            n_estimators=100, 
            max_depth=10, 
            random_state=42,
            n_jobs=-1
        )
        rf_model.fit(X_train, y_train)
        self.models['random_forest'] = rf_model
        
        # 2. XGBoost for better performance
        xgb_model = xgb.XGBRegressor(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            random_state=42,
            n_jobs=-1
        )
        xgb_model.fit(X_train, y_train)
        self.models['xgboost'] = xgb_model
        
        # 3. Gradient Boosting
        gb_model = GradientBoostingRegressor(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            random_state=42
        )
        gb_model.fit(X_train, y_train)
        self.models['gradient_boosting'] = gb_model
        
        # 4. KNN for similarity-based recommendations
        knn_model = NearestNeighbors(n_neighbors=20, metric='cosine')
        knn_model.fit(X_train)
        self.models['knn'] = knn_model
        
        # Evaluate models
        self._evaluate_models(X_test, y_test)
        
    def _train_collaborative_model(self, features_df, target):
        """Train collaborative filtering model"""
        # Use KNN for collaborative filtering
        knn_model = NearestNeighbors(n_neighbors=20, metric='cosine')
        knn_model.fit(features_df)
        self.models['collaborative'] = knn_model
        
    def _train_content_based_model(self, features_df, target):
        """Train content-based filtering model"""
        # Use cosine similarity for content-based filtering
        self.models['content_based'] = cosine_similarity(features_df)
        
    def _train_single_model(self, X_train, y_train, X_test, y_test):
        """Train a single model (XGBoost by default)"""
        model = xgb.XGBRegressor(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            random_state=42
        )
        model.fit(X_train, y_train)
        self.models['single'] = model
        
    def _evaluate_models(self, X_test, y_test):
        """Evaluate model performance"""
        print("\nModel Evaluation:")
        print("-" * 40)
        
        for name, model in self.models.items():
            if name != 'knn':
                try:
                    y_pred = model.predict(X_test)
                    mse = np.mean((y_test - y_pred) ** 2)
                    rmse = np.sqrt(mse)
                    print(f"{name}: RMSE = {rmse:.4f}")
                except:
                    print(f"{name}: Evaluation not applicable")
    
    def get_recommendations(self, user_profile: Dict, top_k: int = 10) -> List[Dict]:
        """Get college recommendations based on user profile"""
        if not self.is_trained:
            raise ValueError("Model must be trained before making recommendations")
        
        # Create user feature vector
        user_features = self._create_user_features(user_profile)
        
        if self.model_type == 'hybrid':
            return self._get_hybrid_recommendations(user_features, top_k)
        elif self.model_type == 'collaborative':
            return self._get_collaborative_recommendations(user_features, top_k)
        elif self.model_type == 'content_based':
            return self._get_content_based_recommendations(user_features, top_k)
        else:
            return self._get_single_model_recommendations(user_features, top_k)
    
    def _create_user_features(self, user_profile: Dict) -> np.ndarray:
        """Create feature vector from user profile"""
        # Initialize feature vector with zeros
        feature_vector = np.zeros(len(self.feature_names))
        
        # Map user profile to features - use processed feature names
        feature_mapping = {
            'gpa': 'avg_gpa',
            'sat_score': 'avg_sat', 
            'act_score': 'avg_act',
            'budget': 'total_cost_in_state'
        }
        
        for user_key, feature_key in feature_mapping.items():
            if user_key in user_profile and feature_key in self.feature_names:
                idx = self.feature_names.index(feature_key)
                feature_vector[idx] = user_profile[user_key]
        
        # Handle categorical features that need encoding
        # For now, we'll skip categorical features in user input to avoid encoding issues
        # The model will use the processed college data for recommendations
        
        # Scale the feature vector
        feature_vector_scaled = self.scaler.transform(feature_vector.reshape(1, -1))
        return feature_vector_scaled.flatten()  # type: ignore
    
    def _get_hybrid_recommendations(self, user_features: np.ndarray, top_k: int) -> List[Dict]:
        """Get recommendations using hybrid approach"""
        if self.college_data is None:
            raise ValueError("College data not loaded")
            
        recommendations = []
        
        # Get predictions from different models
        predictions = {}
        for name, model in self.models.items():
            if name == 'knn':
                # Get nearest neighbors
                distances, indices = model.kneighbors(user_features.reshape(1, -1))
                predictions[name] = indices[0]
            elif name in ['random_forest', 'xgboost', 'gradient_boosting']:
                # Get prediction scores
                scores = model.predict(user_features.reshape(1, -1))
                predictions[name] = scores[0]
        
        # Combine predictions
        college_scores = {}
        
        # Add scores from tree-based models
        for name, score in predictions.items():
            if name in ['random_forest', 'xgboost', 'gradient_boosting']:
                for i, college in self.college_data.iterrows():
                    if i not in college_scores:
                        college_scores[i] = []
                    college_scores[i].append(score)
        
        # Add similarity scores from KNN
        if 'knn' in predictions:
            for idx in predictions['knn']:
                if idx in college_scores:
                    college_scores[idx].append(1.0)  # High similarity
                else:
                    college_scores[idx] = [1.0]
        
        # Calculate final scores
        final_scores = {}
        for college_idx, scores in college_scores.items():
            final_scores[college_idx] = np.mean(scores)
        
        # Sort by score and get top recommendations
        sorted_colleges = sorted(final_scores.items(), key=lambda x: x[1], reverse=True)
        
        for i, (college_idx, score) in enumerate(sorted_colleges[:top_k]):
            # Use raw college data for returning college information
            college = self.raw_college_data.iloc[college_idx]
            recommendation = {
                'rank': i + 1,
                'college_id': int(college['id']),
                'name': college['name'],
                'state': college['state'],
                'confidence_score': round(score, 3),
                'acceptance_rate': college['acceptance_rate'],
                'avg_sat': college['avg_sat'],
                'avg_act': college['avg_act'],
                'avg_gpa': college['avg_gpa'],
                'total_cost_in_state': college['total_cost_in_state'],
                'total_cost_out_state': college['total_cost_out_state'],
                'graduation_rate': college['graduation_rate'],
                'student_population': college['student_population'],
                'campus_size': college['campus_size'],
                'location_type': college['location_type'],
                'public_private': college['public_private'],
                'quality_score': college['quality_score'],
                'affordability_score': college['affordability_score']
            }
            recommendations.append(recommendation)
        
        return recommendations
    
    def _get_collaborative_recommendations(self, user_features: np.ndarray, top_k: int) -> List[Dict]:
        """Get recommendations using collaborative filtering"""
        if self.college_data is None:
            raise ValueError("College data not loaded")
            
        knn_model = self.models['collaborative']
        distances, indices = knn_model.kneighbors(user_features.reshape(1, -1))
        
        recommendations = []
        for i, idx in enumerate(indices[0][:top_k]):
            # Use raw college data for returning college information
            college = self.raw_college_data.iloc[idx]
            recommendation = {
                'rank': i + 1,
                'college_id': int(college['id']),
                'name': college['name'],
                'confidence_score': round(1 - distances[0][i], 3),
                'state': college['state'],
                'acceptance_rate': college['acceptance_rate'],
                'total_cost_in_state': college['total_cost_in_state'],
                'graduation_rate': college['graduation_rate']
            }
            recommendations.append(recommendation)
        
        return recommendations
    
    def _get_content_based_recommendations(self, user_features: np.ndarray, top_k: int) -> List[Dict]:
        """Get recommendations using content-based filtering"""
        if self.college_data is None:
            raise ValueError("College data not loaded")
            
        # Calculate similarity with all colleges
        similarities = cosine_similarity(user_features.reshape(1, -1), self.models['content_based'])
        
        # Get top similar colleges
        top_indices = np.argsort(similarities[0])[::-1][:top_k]
        
        recommendations = []
        for i, idx in enumerate(top_indices):
            # Use raw college data for returning college information
            college = self.raw_college_data.iloc[idx]
            recommendation = {
                'rank': i + 1,
                'college_id': int(college['id']),
                'name': college['name'],
                'confidence_score': round(similarities[0][idx], 3),
                'state': college['state'],
                'acceptance_rate': college['acceptance_rate'],
                'total_cost_in_state': college['total_cost_in_state']
            }
            recommendations.append(recommendation)
        
        return recommendations
    
    def _get_single_model_recommendations(self, user_features: np.ndarray, top_k: int) -> List[Dict]:
        """Get recommendations using single model"""
        if self.college_data is None:
            raise ValueError("College data not loaded")
            
        model = self.models['single']
        predictions = model.predict(user_features.reshape(1, -1))
        
        # Get all college predictions
        all_predictions = model.predict(self.scaler.transform(self.college_data[self.feature_names]))
        
        # Get top predictions
        top_indices = np.argsort(all_predictions)[::-1][:top_k]
        
        recommendations = []
        for i, idx in enumerate(top_indices):
            # Use raw college data for returning college information
            college = self.raw_college_data.iloc[idx]
            recommendation = {
                'rank': i + 1,
                'college_id': int(college['id']),
                'name': college['name'],
                'confidence_score': round(all_predictions[idx], 3),
                'state': college['state'],
                'acceptance_rate': college['acceptance_rate'],
                'total_cost_in_state': college['total_cost_in_state']
            }
            recommendations.append(recommendation)
        
        return recommendations
    
    def save_model(self, filepath: str):
        """Save the trained model"""
        model_data = {
            'model_type': self.model_type,
            'models': self.models,
            'scaler': self.scaler,
            'feature_names': self.feature_names,
            'is_trained': self.is_trained,
            'raw_college_data': self.raw_college_data
        }
        joblib.dump(model_data, filepath)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load a trained model"""
        if os.path.exists(filepath):
            model_data = joblib.load(filepath)
            self.model_type = model_data['model_type']
            self.models = model_data['models']
            self.scaler = model_data['scaler']
            self.feature_names = model_data['feature_names']
            self.is_trained = model_data['is_trained']
            self.raw_college_data = model_data.get('raw_college_data', None)
            print(f"Model loaded from {filepath}")
        else:
            raise FileNotFoundError(f"Model file {filepath} not found") 