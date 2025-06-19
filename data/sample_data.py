import pandas as pd
import numpy as np
import random
import sys
import os

def generate_college_data(num_colleges=500):
    """
    Generate sample college data for the recommendation system.
    This simulates a real college dataset with various features.
    """
    
    # College names (mix of real and fictional)
    college_names = [
        "University of California, Berkeley", "Stanford University", "MIT", "Harvard University",
        "Yale University", "Princeton University", "Columbia University", "University of Chicago",
        "Northwestern University", "Duke University", "University of Pennsylvania", "Cornell University",
        "Brown University", "Dartmouth College", "Vanderbilt University", "Rice University",
        "Washington University in St. Louis", "Emory University", "Georgetown University", "University of Notre Dame",
        "University of Virginia", "University of North Carolina", "Wake Forest University", "Boston College",
        "Tufts University", "Brandeis University", "Case Western Reserve University", "Northeastern University",
        "University of Rochester", "Boston University", "Lehigh University", "Villanova University",
        "University of Miami", "Syracuse University", "University of Pittsburgh", "Penn State University",
        "Ohio State University", "University of Michigan", "University of Wisconsin", "University of Illinois",
        "Purdue University", "Indiana University", "University of Minnesota", "University of Iowa",
        "University of Texas", "Texas A&M University", "Baylor University", "Southern Methodist University",
        "University of Arizona", "Arizona State University", "University of Colorado", "University of Utah",
        "University of Washington", "University of Oregon", "University of California, Los Angeles",
        "University of California, San Diego", "University of California, Davis", "University of California, Irvine"
    ]
    
    # States
    states = [
        "CA", "MA", "NY", "CT", "NJ", "PA", "IL", "MI", "OH", "IN", "WI", "MN", "IA", "TX", "AZ", "CO", "UT", "WA", "OR"
    ]
    
    # Majors
    majors = [
        "Computer Science", "Engineering", "Business", "Biology", "Chemistry", "Physics", "Mathematics",
        "Psychology", "Economics", "Political Science", "History", "English", "Art", "Music", "Medicine",
        "Law", "Education", "Nursing", "Architecture", "Environmental Science"
    ]
    
    # Campus sizes
    campus_sizes = ["Small", "Medium", "Large"]
    
    # Generate data
    data = []
    
    for i in range(num_colleges):
        # Academic features
        acceptance_rate = random.uniform(0.05, 0.95)
        avg_sat = random.randint(1000, 1600)
        avg_act = random.randint(18, 36)
        avg_gpa = random.uniform(2.5, 4.0)
        
        # Financial features
        tuition_in_state = random.randint(5000, 60000)
        tuition_out_state = random.randint(15000, 80000)
        room_board = random.randint(8000, 20000)
        total_cost_in_state = tuition_in_state + room_board
        total_cost_out_state = tuition_out_state + room_board
        
        # Location and size
        state = random.choice(states)
        campus_size = random.choice(campus_sizes)
        student_population = random.randint(1000, 50000)
        
        # Academic quality
        graduation_rate = random.uniform(0.6, 0.98)
        retention_rate = random.uniform(0.7, 0.98)
        
        # Selectivity score (combination of acceptance rate and test scores)
        selectivity_score = (1 - acceptance_rate) * 0.6 + (avg_sat / 1600) * 0.4
        
        # Affordability score (inverse of cost)
        affordability_score = 1 - (total_cost_in_state / 80000)
        
        # Overall quality score
        quality_score = (selectivity_score * 0.4 + graduation_rate * 0.3 + retention_rate * 0.3)
        
        college = {
            'id': i + 1,
            'name': random.choice(college_names) + f" ({state})" if random.random() < 0.3 else random.choice(college_names),
            'state': state,
            'acceptance_rate': round(acceptance_rate, 3),
            'avg_sat': avg_sat,
            'avg_act': avg_act,
            'avg_gpa': round(avg_gpa, 2),
            'tuition_in_state': tuition_in_state,
            'tuition_out_state': tuition_out_state,
            'room_board': room_board,
            'total_cost_in_state': total_cost_in_state,
            'total_cost_out_state': total_cost_out_state,
            'campus_size': campus_size,
            'student_population': student_population,
            'graduation_rate': round(graduation_rate, 3),
            'retention_rate': round(retention_rate, 3),
            'selectivity_score': round(selectivity_score, 3),
            'affordability_score': round(affordability_score, 3),
            'quality_score': round(quality_score, 3),
            'top_majors': random.sample(majors, random.randint(3, 8)),
            'location_type': random.choice(['Urban', 'Suburban', 'Rural']),
            'public_private': random.choice(['Public', 'Private']),
            'religious_affiliation': random.choice(['None', 'Catholic', 'Protestant', 'Jewish', 'Other']),
            'athletics_division': random.choice(['NCAA Division I', 'NCAA Division II', 'NCAA Division III', 'NAIA', 'None'])
        }
        
        data.append(college)
    
    return pd.DataFrame(data)

def save_sample_data():
    """Generate and save sample college data to CSV"""
    df = generate_college_data(500)
    df.to_csv('data/colleges.csv', index=False)
    print(f"Generated {len(df)} college records and saved to data/colleges.csv")
    return df

if __name__ == "__main__":
    save_sample_data() 