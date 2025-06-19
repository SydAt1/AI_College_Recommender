from pydantic import BaseModel, Field, validator
from typing import List, Optional, Dict, Any
from enum import Enum

class CampusSize(str, Enum):
    SMALL = "Small"
    MEDIUM = "Medium"
    LARGE = "Large"

class LocationType(str, Enum):
    URBAN = "Urban"
    SUBURBAN = "Suburban"
    RURAL = "Rural"

class PublicPrivate(str, Enum):
    PUBLIC = "Public"
    PRIVATE = "Private"

class UserProfile(BaseModel):
    """User profile for college recommendations"""
    
    # Academic information
    gpa: float = Field(..., ge=0.0, le=4.0, description="Grade Point Average (0.0-4.0)")
    sat_score: Optional[int] = Field(None, ge=400, le=1600, description="SAT Score (400-1600)")
    act_score: Optional[int] = Field(None, ge=1, le=36, description="ACT Score (1-36)")
    class_rank_percentile: Optional[float] = Field(None, ge=0.0, le=100.0, description="Class rank percentile")
    
    # Preferences
    preferred_states: Optional[List[str]] = Field(None, description="Preferred states (2-letter codes)")
    campus_size_preference: Optional[CampusSize] = Field(None, description="Preferred campus size")
    location_type_preference: Optional[LocationType] = Field(None, description="Preferred location type")
    public_private_preference: Optional[PublicPrivate] = Field(None, description="Public or private preference")
    
    # Financial information
    budget_max: Optional[float] = Field(None, ge=0, description="Maximum budget for college costs")
    need_financial_aid: Optional[bool] = Field(None, description="Whether financial aid is needed")
    
    # Academic interests
    intended_major: Optional[str] = Field(None, description="Intended major of study")
    academic_interests: Optional[List[str]] = Field(None, description="List of academic interests")
    
    # Extracurricular activities
    extracurricular_activities: Optional[List[str]] = Field(None, description="List of extracurricular activities")
    leadership_roles: Optional[List[str]] = Field(None, description="Leadership roles held")
    community_service_hours: Optional[int] = Field(None, ge=0, description="Community service hours")
    
    # Additional preferences
    religious_affiliation: Optional[str] = Field(None, description="Religious affiliation preference")
    athletics_interest: Optional[bool] = Field(None, description="Interest in athletics")
    study_abroad_interest: Optional[bool] = Field(None, description="Interest in study abroad")
    
    @validator('sat_score', 'act_score')
    def validate_test_scores(cls, v):
        if v is not None:
            if isinstance(v, str):
                try:
                    v = int(v)
                except ValueError:
                    raise ValueError("Test score must be a valid integer")
        return v
    
    @validator('preferred_states')
    def validate_states(cls, v):
        if v is not None:
            valid_states = [
                'AL', 'AK', 'AZ', 'AR', 'CA', 'CO', 'CT', 'DE', 'FL', 'GA',
                'HI', 'ID', 'IL', 'IN', 'IA', 'KS', 'KY', 'LA', 'ME', 'MD',
                'MA', 'MI', 'MN', 'MS', 'MO', 'MT', 'NE', 'NV', 'NH', 'NJ',
                'NM', 'NY', 'NC', 'ND', 'OH', 'OK', 'OR', 'PA', 'RI', 'SC',
                'SD', 'TN', 'TX', 'UT', 'VT', 'VA', 'WA', 'WV', 'WI', 'WY'
            ]
            for state in v:
                if state.upper() not in valid_states:
                    raise ValueError(f"Invalid state code: {state}")
            return [state.upper() for state in v]
        return v

class CollegeRecommendation(BaseModel):
    """Individual college recommendation"""
    
    rank: int = Field(..., description="Ranking position")
    college_id: int = Field(..., description="Unique college identifier")
    name: str = Field(..., description="College name")
    state: str = Field(..., description="State location")
    confidence_score: float = Field(..., ge=0.0, le=1.0, description="Confidence score (0-1)")
    
    # Academic information
    acceptance_rate: float = Field(..., description="Acceptance rate")
    avg_sat: int = Field(..., description="Average SAT score")
    avg_act: int = Field(..., description="Average ACT score")
    avg_gpa: float = Field(..., description="Average GPA")
    graduation_rate: float = Field(..., description="Graduation rate")
    
    # Financial information
    total_cost_in_state: int = Field(..., description="Total cost for in-state students")
    total_cost_out_state: int = Field(..., description="Total cost for out-of-state students")
    
    # Campus information
    student_population: int = Field(..., description="Student population")
    campus_size: str = Field(..., description="Campus size category")
    location_type: str = Field(..., description="Location type")
    public_private: str = Field(..., description="Public or private institution")
    
    # Quality scores
    quality_score: float = Field(..., description="Overall quality score")
    affordability_score: float = Field(..., description="Affordability score")

class RecommendationResponse(BaseModel):
    """API response for college recommendations"""
    
    recommendations: List[CollegeRecommendation] = Field(..., description="List of college recommendations")
    total_recommendations: int = Field(..., description="Total number of recommendations")
    user_profile: UserProfile = Field(..., description="User profile used for recommendations")
    model_info: Dict[str, Any] = Field(..., description="Information about the model used")
    processing_time: float = Field(..., description="Time taken to generate recommendations")

class HealthCheck(BaseModel):
    """Health check response"""
    
    status: str = Field(..., description="Service status")
    model_loaded: bool = Field(..., description="Whether the ML model is loaded")
    total_colleges: int = Field(..., description="Total number of colleges in database")
    version: str = Field(..., description="API version")

class ErrorResponse(BaseModel):
    """Error response"""
    
    error: str = Field(..., description="Error message")
    detail: Optional[str] = Field(None, description="Detailed error information")
    status_code: int = Field(..., description="HTTP status code") 