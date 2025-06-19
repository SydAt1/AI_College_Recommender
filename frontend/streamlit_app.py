import streamlit as st
import requests
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
from datetime import datetime
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.report_generator import generate_pdf_report, generate_csv_report
from utils.visualizations import create_recommendation_charts

# Page configuration
st.set_page_config(
    page_title="AI College Recommender",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        color: #1f77b4;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        font-weight: bold;
        color: #2c3e50;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .recommendation-card {
        background-color: white;
        padding: 1.5rem;
        border-radius: 0.5rem;
        border: 1px solid #e0e0e0;
        margin-bottom: 1rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .confidence-high { color: #28a745; }
    .confidence-medium { color: #ffc107; }
    .confidence-low { color: #dc3545; }
</style>
""", unsafe_allow_html=True)

# API configuration
API_BASE_URL = "http://localhost:8000"

def check_api_health():
    """Check if the API is running"""
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=5)
        return response.status_code == 200, response.json()
    except:
        return False, None

def get_recommendations(user_profile, top_k=10):
    """Get recommendations from the API"""
    try:
        response = requests.post(
            f"{API_BASE_URL}/recommend",
            json=user_profile,
            params={"top_k": top_k},
            timeout=30
        )
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"API Error: {response.status_code} - {response.text}")
            return None
    except Exception as e:
        st.error(f"Error connecting to API: {str(e)}")
        return None

def get_college_stats():
    """Get college database statistics"""
    try:
        response = requests.get(f"{API_BASE_URL}/stats", timeout=10)
        if response.status_code == 200:
            return response.json()
        return None
    except:
        return None

def main():
    # Header
    st.markdown('<h1 class="main-header">🎓 AI College Recommender</h1>', unsafe_allow_html=True)
    st.markdown("### Find your perfect college match using artificial intelligence")
    
    # Check API health
    api_healthy, health_data = check_api_health()
    
    if not api_healthy:
        st.error("⚠️ **API Connection Error**")
        st.warning("""
        The AI College Recommender API is not running. Please start the API server first:
        
        ```bash
        cd api
        uvicorn main:app --reload
        ```
        
        Then refresh this page.
        """)
        return
    
    # Sidebar
    with st.sidebar:
        st.markdown("### 🎯 Quick Stats")
        stats = get_college_stats()
        if stats:
            st.metric("Total Colleges", f"{stats['total_colleges']:,}")
            st.metric("States Covered", stats['states_represented'])
            st.metric("Avg Acceptance Rate", f"{stats['average_acceptance_rate']:.1%}")
            st.metric("Avg Cost (In-State)", f"${stats['average_cost_in_state']:,}")
        
        st.markdown("---")
        st.markdown("### 📊 Model Status")
        if health_data:
            status_color = "🟢" if health_data['model_loaded'] else "🔴"
            st.write(f"{status_color} Model: {'Loaded' if health_data['model_loaded'] else 'Not Loaded'}")
            st.write(f"📈 Version: {health_data['version']}")
    
    # Main content
    tab1, tab2, tab3, tab4 = st.tabs(["🎯 Get Recommendations", "📊 Explore Colleges", "📈 Analytics", "📋 About"])
    
    with tab1:
        st.markdown('<h2 class="sub-header">Get Your Personalized College Recommendations</h2>', unsafe_allow_html=True)
        
        # User input form
        with st.form("recommendation_form"):
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### 📚 Academic Information")
                gpa = st.slider("GPA", 0.0, 4.0, 3.5, 0.1, help="Your cumulative Grade Point Average")
                
                sat_score = st.number_input("SAT Score", 400, 1600, 1200, 10, 
                                          help="Your SAT score (leave 0 if not taken)")
                if sat_score == 0:
                    sat_score = None
                
                act_score = st.number_input("ACT Score", 1, 36, 25, 1,
                                          help="Your ACT score (leave 0 if not taken)")
                if act_score == 0:
                    act_score = None
                
                class_rank = st.slider("Class Rank Percentile", 0.0, 100.0, 75.0, 1.0,
                                     help="Your class rank percentile (100 = top of class)")
            
            with col2:
                st.markdown("#### 🎯 Preferences")
                preferred_states = st.multiselect(
                    "Preferred States",
                    ["CA", "NY", "TX", "FL", "PA", "IL", "OH", "MI", "NC", "GA", "VA", "WA", "OR", "CO", "AZ"],
                    help="Select your preferred states"
                )
                
                campus_size = st.selectbox(
                    "Preferred Campus Size",
                    ["Small", "Medium", "Large"],
                    help="Choose your preferred campus size"
                )
                
                location_type = st.selectbox(
                    "Preferred Location Type",
                    ["Urban", "Suburban", "Rural"],
                    help="Choose your preferred location type"
                )
                
                public_private = st.selectbox(
                    "Public or Private",
                    ["Public", "Private"],
                    help="Choose between public and private institutions"
                )
            
            st.markdown("#### 💰 Financial Information")
            col3, col4 = st.columns(2)
            
            with col3:
                budget_max = st.number_input("Maximum Budget ($)", 0, 100000, 50000, 1000,
                                           help="Your maximum budget for college costs")
                need_financial_aid = st.checkbox("Need Financial Aid", help="Check if you need financial aid")
            
            with col4:
                intended_major = st.text_input("Intended Major", help="Your intended major of study")
                top_k = st.slider("Number of Recommendations", 5, 20, 10, 1,
                                help="Number of colleges to recommend")
            
            submitted = st.form_submit_button("🚀 Get Recommendations", use_container_width=True)
        
        if submitted:
            with st.spinner("🤖 AI is analyzing your profile and finding the best colleges..."):
                # Create user profile
                user_profile = {
                    "gpa": gpa,
                    "sat_score": sat_score,
                    "act_score": act_score,
                    "class_rank_percentile": class_rank,
                    "preferred_states": preferred_states,
                    "campus_size_preference": campus_size,
                    "location_type_preference": location_type,
                    "public_private_preference": public_private,
                    "budget_max": budget_max,
                    "need_financial_aid": need_financial_aid,
                    "intended_major": intended_major if intended_major else None
                }
                
                # Get recommendations
                result = get_recommendations(user_profile, top_k)
                
                if result:
                    st.success(f"✅ Found {result['total_recommendations']} recommendations in {result['processing_time']} seconds!")
                    
                    # Display recommendations
                    st.markdown("### 🎓 Your Recommended Colleges")
                    
                    for i, rec in enumerate(result['recommendations']):
                        with st.container():
                            col1, col2, col3 = st.columns([2, 1, 1])
                            
                            with col1:
                                st.markdown(f"**{rec['rank']}. {rec['name']}**")
                                st.write(f"📍 {rec['state']} • {rec['campus_size']} • {rec['location_type']}")
                                st.write(f"🎯 Acceptance Rate: {rec['acceptance_rate']:.1%} • Graduation Rate: {rec['graduation_rate']:.1%}")
                            
                            with col2:
                                confidence = rec['confidence_score']
                                if confidence >= 0.7:
                                    confidence_class = "confidence-high"
                                elif confidence >= 0.4:
                                    confidence_class = "confidence-medium"
                                else:
                                    confidence_class = "confidence-low"
                                
                                st.markdown(f"<span class='{confidence_class}'><strong>Match Score: {confidence:.1%}</strong></span>", 
                                           unsafe_allow_html=True)
                                st.write(f"💰 In-State: ${rec['total_cost_in_state']:,}")
                                st.write(f"💰 Out-State: ${rec['total_cost_out_state']:,}")
                            
                            with col3:
                                st.write(f"📊 SAT: {rec['avg_sat']}")
                                st.write(f"📊 ACT: {rec['avg_act']}")
                                st.write(f"📊 GPA: {rec['avg_gpa']}")
                            
                            st.markdown("---")
                    
                    # Download options
                    st.markdown("### 📥 Download Your Recommendations")
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        if st.button("📄 Download PDF Report", use_container_width=True):
                            pdf_bytes = generate_pdf_report(result, user_profile)
                            st.download_button(
                                label="📄 Download PDF",
                                data=pdf_bytes,
                                file_name=f"college_recommendations_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                                mime="application/pdf"
                            )
                    
                    with col2:
                        if st.button("📊 Download CSV Report", use_container_width=True):
                            csv_data = generate_csv_report(result, user_profile)
                            st.download_button(
                                label="📊 Download CSV",
                                data=csv_data,
                                file_name=f"college_recommendations_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                                mime="text/csv"
                            )
    
    with tab2:
        st.markdown('<h2 class="sub-header">Explore College Database</h2>', unsafe_allow_html=True)
        
        # Filters
        col1, col2, col3 = st.columns(3)
        
        with col1:
            state_filter = st.selectbox("Filter by State", ["All"] + ["CA", "NY", "TX", "FL", "PA", "IL", "OH", "MI", "NC", "GA", "VA", "WA", "OR", "CO", "AZ"])
        
        with col2:
            size_filter = st.selectbox("Filter by Size", ["All", "Small", "Medium", "Large"])
        
        with col3:
            type_filter = st.selectbox("Filter by Type", ["All", "Public", "Private"])
        
        # Get colleges
        try:
            params = {}
            if state_filter != "All":
                params["state"] = state_filter
            if size_filter != "All":
                params["campus_size"] = size_filter
            if type_filter != "All":
                params["public_private"] = type_filter
            
            response = requests.get(f"{API_BASE_URL}/colleges", params=params, timeout=10)
            if response.status_code == 200:
                colleges = response.json()
                
                if colleges:
                    df = pd.DataFrame(colleges)
                    
                    # Display colleges
                    st.markdown(f"### 📊 Found {len(colleges)} Colleges")
                    
                    # Create a searchable dataframe
                    search_term = st.text_input("🔍 Search colleges by name:")
                    if search_term:
                        df = df[df['name'].str.contains(search_term, case=False)]
                    
                    # Display as a table
                    st.dataframe(
                        df[['name', 'state', 'acceptance_rate', 'avg_sat', 'total_cost_in_state', 'graduation_rate']],
                        use_container_width=True
                    )
                    
                    # Show detailed view for selected college
                    if len(df) > 0:
                        selected_college = st.selectbox("Select a college for detailed view:", df['name'].tolist())
                        if selected_college:
                            college_id = df[df['name'] == selected_college]['id'].iloc[0]  # type: ignore
                            
                            # Get detailed information
                            detail_response = requests.get(f"{API_BASE_URL}/colleges/{college_id}", timeout=10)
                            if detail_response.status_code == 200:
                                college_detail = detail_response.json()
                                
                                st.markdown("### 📋 College Details")
                                col1, col2 = st.columns(2)
                                
                                with col1:
                                    st.write(f"**Name:** {college_detail['name']}")
                                    st.write(f"**State:** {college_detail['state']}")
                                    st.write(f"**Type:** {college_detail['public_private']}")
                                    st.write(f"**Campus Size:** {college_detail['campus_size']}")
                                    st.write(f"**Location:** {college_detail['location_type']}")
                                
                                with col2:
                                    st.write(f"**Acceptance Rate:** {college_detail['acceptance_rate']:.1%}")
                                    st.write(f"**Graduation Rate:** {college_detail['graduation_rate']:.1%}")
                                    st.write(f"**Student Population:** {college_detail['student_population']:,}")
                                    st.write(f"**Quality Score:** {college_detail['quality_score']:.3f}")
                                    st.write(f"**Affordability Score:** {college_detail['affordability_score']:.3f}")
                
                else:
                    st.warning("No colleges found with the selected filters.")
            else:
                st.error("Error fetching colleges from API")
                
        except Exception as e:
            st.error(f"Error: {str(e)}")
    
    with tab3:
        st.markdown('<h2 class="sub-header">Analytics & Insights</h2>', unsafe_allow_html=True)
        
        if stats:
            # Key metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Colleges", f"{stats['total_colleges']:,}")
            
            with col2:
                st.metric("States Covered", stats['states_represented'])
            
            with col3:
                st.metric("Public Colleges", stats['public_colleges'])
            
            with col4:
                st.metric("Private Colleges", stats['private_colleges'])
            
            # Charts
            col1, col2 = st.columns(2)
            
            with col1:
                # Campus size distribution
                try:
                    if stats.get('campus_size_distribution') and len(stats['campus_size_distribution']) > 0:
                        size_items = list(stats['campus_size_distribution'].items())
                        size_data = pd.DataFrame(size_items, columns=['Size', 'Count'])  # type: ignore
                        fig_size = px.pie(size_data, values='Count', names='Size', 
                                        title="Campus Size Distribution")
                        st.plotly_chart(fig_size, use_container_width=True)
                    else:
                        st.info("No campus size distribution data available")
                except Exception as e:
                    st.warning(f"Could not display campus size distribution: {str(e)}")
            
            with col2:
                # Location type distribution
                try:
                    if stats.get('location_type_distribution') and len(stats['location_type_distribution']) > 0:
                        location_items = list(stats['location_type_distribution'].items())
                        location_data = pd.DataFrame(location_items, columns=['Location', 'Count'])  # type: ignore
                        fig_location = px.bar(location_data, x='Location', y='Count', 
                                            title="Location Type Distribution")
                        st.plotly_chart(fig_location, use_container_width=True)
                    else:
                        st.info("No location type distribution data available")
                except Exception as e:
                    st.warning(f"Could not display location type distribution: {str(e)}")
            
            # Cost vs Quality scatter plot
            try:
                response = requests.get(f"{API_BASE_URL}/colleges", params={"limit": 100}, timeout=10)
                if response.status_code == 200:
                    colleges = response.json()
                    df = pd.DataFrame(colleges)
                    
                    fig_scatter = px.scatter(
                        df, x='total_cost_in_state', y='quality_score',
                        hover_data=['name', 'state'],
                        title="Cost vs Quality Analysis",
                        labels={'total_cost_in_state': 'Total Cost (In-State)', 'quality_score': 'Quality Score'}
                    )
                    st.plotly_chart(fig_scatter, use_container_width=True)
            except:
                st.warning("Could not load scatter plot data")
    
    with tab4:
        st.markdown('<h2 class="sub-header">About AI College Recommender</h2>', unsafe_allow_html=True)
        
        st.markdown("""
        ### 🎯 What is AI College Recommender?
        
        AI College Recommender is an intelligent system that uses machine learning to help students find their perfect college match. 
        Our advanced algorithms analyze your academic profile, preferences, and financial situation to provide personalized recommendations.
        
        ### 🤖 How It Works
        
        1. **Data Analysis**: We analyze comprehensive college data including academic metrics, costs, and student outcomes
        2. **ML Models**: Our hybrid recommendation system combines multiple AI approaches:
           - **Collaborative Filtering**: Finds colleges similar students chose
           - **Content-Based Filtering**: Matches your preferences with college features
           - **Hybrid Approach**: Combines both methods for optimal results
        3. **Personalization**: Your unique profile is matched against our database
        4. **Ranking**: Colleges are ranked by confidence score and relevance
        
        ### 📊 Features
        
        - **Smart Recommendations**: ML-powered college matching
        - **Comprehensive Data**: 500+ colleges with detailed information
        - **Multiple Criteria**: Academic, financial, and preference-based matching
        - **Confidence Scoring**: Each recommendation comes with a confidence score
        - **Export Options**: Download recommendations as PDF or CSV
        - **Interactive Analytics**: Visualize college data and trends
        
        ### 🛠️ Technology Stack
        
        - **Backend**: FastAPI with Pydantic validation
        - **ML Models**: scikit-learn, XGBoost, and custom hybrid algorithms
        - **Frontend**: Streamlit with Plotly visualizations
        - **Data Processing**: Advanced feature engineering and preprocessing
        
        ### 📈 Model Performance
        
        Our hybrid recommendation system achieves high accuracy by combining:
        - Random Forest for feature importance
        - XGBoost for prediction accuracy
        - K-Nearest Neighbors for similarity matching
        - Gradient Boosting for ensemble learning
        
        ### 🔒 Privacy & Security
        
        - Your data is processed locally and not stored
        - No personal information is shared with third parties
        - All API communications are secure
        
        ### 📞 Support
        
        For questions or support, please refer to the project documentation or contact the development team.
        """)

if __name__ == "__main__":
    main() 