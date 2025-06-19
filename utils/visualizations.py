import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
from typing import List, Dict, Any

def create_recommendation_charts(recommendations: List[Dict[str, Any]]) -> Dict[str, go.Figure]:
    """Create various charts for recommendation analysis"""
    
    if not recommendations:
        return {}
    
    df = pd.DataFrame(recommendations)
    charts = {}
    
    # 1. Confidence Score Distribution
    fig_confidence = px.histogram(
        df, x='confidence_score',
        title="Distribution of Confidence Scores",
        labels={'confidence_score': 'Confidence Score', 'count': 'Number of Colleges'},
        nbins=10
    )
    fig_confidence.update_layout(
        xaxis_title="Confidence Score",
        yaxis_title="Number of Colleges",
        showlegend=False
    )
    charts['confidence_distribution'] = fig_confidence
    
    # 2. Cost vs Quality Scatter Plot
    fig_scatter = px.scatter(
        df, x='total_cost_in_state', y='quality_score',
        hover_data=['name', 'state', 'confidence_score'],
        title="Cost vs Quality Analysis",
        labels={
            'total_cost_in_state': 'Total Cost (In-State)',
            'quality_score': 'Quality Score'
        },
        color='confidence_score',
        color_continuous_scale='viridis'
    )
    fig_scatter.update_layout(
        xaxis_title="Total Cost (In-State) $",
        yaxis_title="Quality Score"
    )
    charts['cost_vs_quality'] = fig_scatter
    
    # 3. Acceptance Rate vs Graduation Rate
    fig_acceptance_grad = px.scatter(
        df, x='acceptance_rate', y='graduation_rate',
        hover_data=['name', 'state'],
        title="Acceptance Rate vs Graduation Rate",
        labels={
            'acceptance_rate': 'Acceptance Rate',
            'graduation_rate': 'Graduation Rate'
        },
        color='confidence_score',
        color_continuous_scale='plasma'
    )
    fig_acceptance_grad.update_layout(
        xaxis_title="Acceptance Rate",
        yaxis_title="Graduation Rate"
    )
    charts['acceptance_vs_graduation'] = fig_acceptance_grad
    
    # 4. State Distribution
    state_counts = df['state'].value_counts()
    fig_states = px.bar(
        x=state_counts.index,
        y=state_counts.values,
        title="Colleges by State",
        labels={'x': 'State', 'y': 'Number of Colleges'}
    )
    fig_states.update_layout(
        xaxis_title="State",
        yaxis_title="Number of Colleges",
        showlegend=False
    )
    charts['state_distribution'] = fig_states
    
    # 5. Campus Size Distribution
    size_counts = df['campus_size'].value_counts()
    fig_sizes = px.pie(
        values=size_counts.values,
        names=size_counts.index,
        title="Distribution by Campus Size"
    )
    charts['campus_size_distribution'] = fig_sizes
    
    # 6. Public vs Private Distribution
    type_counts = df['public_private'].value_counts()
    fig_types = px.pie(
        values=type_counts.values,
        names=type_counts.index,
        title="Public vs Private Institutions"
    )
    charts['institution_type_distribution'] = fig_types
    
    # 7. Top Colleges by Confidence Score
    top_colleges = df.nlargest(10, 'confidence_score')
    fig_top = px.bar(
        top_colleges, x='name', y='confidence_score',
        title="Top 10 Colleges by Confidence Score",
        labels={'name': 'College Name', 'confidence_score': 'Confidence Score'}
    )
    fig_top.update_layout(
        xaxis_title="College Name",
        yaxis_title="Confidence Score",
        xaxis_tickangle=-45
    )
    charts['top_colleges'] = fig_top
    
    return charts 