from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import time
import sys
import os
from typing import List, Dict, Any
import uvicorn
import pandas as pd

# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.college_model import CollegeRecommendationModel
from models.data_processor import CollegeDataProcessor
from api.schemas import (
    UserProfile, CollegeRecommendation, RecommendationResponse, 
    HealthCheck, ErrorResponse
)

# Initialize FastAPI app
app = FastAPI(
    title="AI College Recommender API",
    description="An intelligent API for college recommendations using machine learning",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify actual origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for model and data
model = None
data_processor = None
college_data = None

@app.on_event("startup")
async def startup_event():
    """Initialize the model and data on startup"""
    global model, data_processor, college_data
    
    try:
        print("Initializing AI College Recommender...")
        
        # Initialize data processor
        data_processor = CollegeDataProcessor()
        
        # Load and process data
        college_data = data_processor.load_data()
        if college_data is None:
            print("Error: Could not load college data")
            return
        
        # Prepare features
        features_df, processed_data = data_processor.prepare_features(college_data)
        
        # Initialize and train model with both raw and processed data
        model = CollegeRecommendationModel(model_type='hybrid')
        model.train(features_df, processed_data, raw_college_data=college_data)
        
        print("✅ AI College Recommender initialized successfully!")
        print(f"📊 Loaded {len(college_data)} colleges")
        print(f"🔧 Model type: {model.model_type}")
        
    except Exception as e:
        print(f"❌ Error initializing model: {str(e)}")
        raise e

def get_model():
    """Dependency to get the trained model"""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return model

def get_data_processor():
    """Dependency to get the data processor"""
    if data_processor is None:
        raise HTTPException(status_code=503, detail="Data processor not loaded")
    return data_processor

@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint"""
    return {
        "message": "AI College Recommender API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health"
    }

@app.get("/health", response_model=HealthCheck)
async def health_check():
    """Health check endpoint"""
    return HealthCheck(
        status="healthy" if model is not None else "unhealthy",
        model_loaded=model is not None and model.is_trained,
        total_colleges=len(college_data) if college_data is not None else 0,
        version="1.0.0"
    )

@app.post("/recommend", response_model=RecommendationResponse)
async def get_recommendations(
    user_profile: UserProfile,
    top_k: int = 10,
    model_instance: CollegeRecommendationModel = Depends(get_model)
):
    """Get college recommendations based on user profile"""
    
    if top_k < 1 or top_k > 50:
        raise HTTPException(status_code=400, detail="top_k must be between 1 and 50")
    
    start_time = time.time()
    
    try:
        # Convert user profile to model input format
        model_input = _convert_user_profile_to_model_input(user_profile)
        
        # Get recommendations
        recommendations = model_instance.get_recommendations(model_input, top_k)
        
        # Convert to response format
        college_recommendations = []
        for rec in recommendations:
            college_rec = CollegeRecommendation(
                rank=rec['rank'],
                college_id=rec['college_id'],
                name=rec['name'],
                state=rec['state'],
                confidence_score=rec['confidence_score'],
                acceptance_rate=rec['acceptance_rate'],
                avg_sat=rec['avg_sat'],
                avg_act=rec['avg_act'],
                avg_gpa=rec['avg_gpa'],
                graduation_rate=rec['graduation_rate'],
                total_cost_in_state=rec['total_cost_in_state'],
                total_cost_out_state=rec['total_cost_out_state'],
                student_population=rec['student_population'],
                campus_size=rec['campus_size'],
                location_type=rec['location_type'],
                public_private=rec['public_private'],
                quality_score=rec['quality_score'],
                affordability_score=rec['affordability_score']
            )
            college_recommendations.append(college_rec)
        
        processing_time = time.time() - start_time
        
        return RecommendationResponse(
            recommendations=college_recommendations,
            total_recommendations=len(college_recommendations),
            user_profile=user_profile,
            model_info={
                "model_type": model_instance.model_type,
                "total_colleges": len(college_data) if college_data is not None else 0,
                "features_used": len(model_instance.feature_names)
            },
            processing_time=round(processing_time, 3)
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating recommendations: {str(e)}")

@app.get("/colleges", response_model=List[Dict[str, Any]])
async def get_colleges(
    state: str | None = None,
    campus_size: str | None = None,
    public_private: str | None = None,
    limit: int = 50
):
    """Get list of colleges with optional filtering"""
    
    if college_data is None:
        raise HTTPException(status_code=503, detail="College data not loaded")
    
    if limit < 1 or limit > 100:
        raise HTTPException(status_code=400, detail="limit must be between 1 and 100")
    
    # Filter data
    filtered_data = college_data.copy()
    
    if state:
        filtered_data = filtered_data[filtered_data['state'] == state.upper()]
    
    if campus_size:
        filtered_data = filtered_data[filtered_data['campus_size'] == campus_size]
    
    if public_private:
        filtered_data = filtered_data[filtered_data['public_private'] == public_private]
    
    # Limit results
    filtered_data = filtered_data.head(limit)  # type: ignore
    
    # Convert to list of dictionaries
    colleges = []
    for _, row in filtered_data.iterrows():
        college = {
            'id': int(row['id']),
            'name': row['name'],
            'state': row['state'],
            'acceptance_rate': row['acceptance_rate'],
            'avg_sat': row['avg_sat'],
            'avg_act': row['avg_act'],
            'avg_gpa': row['avg_gpa'],
            'total_cost_in_state': row['total_cost_in_state'],
            'total_cost_out_state': row['total_cost_out_state'],
            'graduation_rate': row['graduation_rate'],
            'student_population': row['student_population'],
            'campus_size': row['campus_size'],
            'location_type': row['location_type'],
            'public_private': row['public_private'],
            'quality_score': row['quality_score'],
            'affordability_score': row['affordability_score']
        }
        colleges.append(college)
    
    return colleges

@app.get("/colleges/{college_id}", response_model=Dict[str, Any])
async def get_college_by_id(college_id: int):
    """Get detailed information about a specific college"""
    
    if college_data is None:
        raise HTTPException(status_code=503, detail="College data not loaded")
    
    college = college_data[college_data['id'] == college_id]
    
    if college.empty:
        raise HTTPException(status_code=404, detail=f"College with ID {college_id} not found")
    
    college_row = college.iloc[0]
    
    return {
        'id': int(college_row['id']),
        'name': college_row['name'],
        'state': college_row['state'],
        'acceptance_rate': college_row['acceptance_rate'],
        'avg_sat': college_row['avg_sat'],
        'avg_act': college_row['avg_act'],
        'avg_gpa': college_row['avg_gpa'],
        'tuition_in_state': college_row['tuition_in_state'],
        'tuition_out_state': college_row['tuition_out_state'],
        'room_board': college_row['room_board'],
        'total_cost_in_state': college_row['total_cost_in_state'],
        'total_cost_out_state': college_row['total_cost_out_state'],
        'graduation_rate': college_row['graduation_rate'],
        'retention_rate': college_row['retention_rate'],
        'student_population': college_row['student_population'],
        'campus_size': college_row['campus_size'],
        'location_type': college_row['location_type'],
        'public_private': college_row['public_private'],
        'religious_affiliation': college_row['religious_affiliation'],
        'athletics_division': college_row['athletics_division'],
        'quality_score': college_row['quality_score'],
        'affordability_score': college_row['affordability_score'],
        'selectivity_score': college_row['selectivity_score'],
        'top_majors': college_row['top_majors'] if 'top_majors' in college_row else []
    }

@app.get("/stats", response_model=Dict[str, Any])
async def get_statistics():
    """Get statistics about the college database"""
    
    if college_data is None:
        raise HTTPException(status_code=503, detail="College data not loaded")
    
    return {
        'total_colleges': len(college_data),
        'states_represented': college_data['state'].nunique(),
        'average_acceptance_rate': round(college_data['acceptance_rate'].mean(), 3),
        'average_sat_score': round(college_data['avg_sat'].mean()),
        'average_act_score': round(college_data['avg_act'].mean()),
        'average_gpa': round(college_data['avg_gpa'].mean(), 2),
        'average_cost_in_state': round(college_data['total_cost_in_state'].mean()),
        'average_cost_out_state': round(college_data['total_cost_out_state'].mean()),
        'average_graduation_rate': round(college_data['graduation_rate'].mean(), 3),
        'public_colleges': len(college_data[college_data['public_private'] == 'Public']),
        'private_colleges': len(college_data[college_data['public_private'] == 'Private']),
        'campus_size_distribution': college_data['campus_size'].value_counts().to_dict(),
        'location_type_distribution': college_data['location_type'].value_counts().to_dict()
    }

def _convert_user_profile_to_model_input(user_profile: UserProfile) -> Dict[str, Any]:
    """Convert user profile to model input format"""
    model_input = {}
    
    # Academic information
    if user_profile.gpa is not None:
        model_input['gpa'] = user_profile.gpa
    
    if user_profile.sat_score is not None:
        model_input['sat_score'] = user_profile.sat_score
    
    if user_profile.act_score is not None:
        model_input['act_score'] = user_profile.act_score
    
    # Preferences
    if user_profile.preferred_states:
        model_input['preferred_state'] = user_profile.preferred_states[0]  # Use first preference
    
    if user_profile.campus_size_preference:
        model_input['campus_size_preference'] = user_profile.campus_size_preference.value
    
    if user_profile.location_type_preference:
        model_input['location_type_preference'] = user_profile.location_type_preference.value
    
    # Financial information
    if user_profile.budget_max is not None:
        model_input['budget'] = user_profile.budget_max
    
    return model_input

@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler"""
    return JSONResponse(
        status_code=500,
        content=ErrorResponse(
            error="Internal server error",
            detail=str(exc),
            status_code=500
        ).dict()
    )

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True) 