# AI College Recommender 🎓

An intelligent college recommendation system that helps students find their perfect college match using machine learning.

## 🛠️ Tech Stack

- **ML Model**: scikit-learn & XGBoost
- **API**: FastAPI with Pydantic validation
- **Frontend**: Streamlit
- **Visualization**: Plotly
- **Report Generation**: PDF/CSV exports

## 🚀 Features

- **Smart Recommendations**: ML-powered college matching based on student preferences
- **Interactive UI**: User-friendly Streamlit interface with sliders and dropdowns
- **Confidence Scoring**: Each recommendation comes with a confidence score
- **Detailed Information**: Location, cost, acceptance rate, and more
- **Export Reports**: Download recommendations as PDF or CSV
- **Visual Comparisons**: Plotly charts for college comparisons

## 📁 Project Structure

```
ai-college-recommend/
├── data/
│   ├── colleges.csv
│   └── sample_data.py
├── models/
│   ├── __init__.py
│   ├── college_model.py
│   └── data_processor.py
├── api/
│   ├── __init__.py
│   ├── main.py
│   └── schemas.py
├── frontend/
│   ├── streamlit_app.py
│   └── components.py
├── utils/
│   ├── __init__.py
│   ├── report_generator.py
│   └── visualizations.py
├── requirements.txt
└── README.md
```

## 🏃‍♂️ Quick Start

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the API Backend**:
   ```bash
   cd api
   uvicorn main:app --reload
   ```

3. **Run the Streamlit Frontend**:
   ```bash
   cd frontend
   streamlit run streamlit_app.py
   ```

4. **Access the Application**:
   - Frontend: http://localhost:8501
   - API Docs: http://localhost:8000/docs

## 📊 How It Works

1. **Data Processing**: The system uses a comprehensive college dataset with features like location, cost, acceptance rate, etc.
2. **ML Model**: Trains a recommendation model using scikit-learn/XGBoost based on user preferences
3. **API Layer**: FastAPI serves predictions with input validation
4. **Frontend**: Streamlit provides an intuitive interface for user input
5. **Reports**: Generate downloadable reports with recommendations and visualizations

## 🎯 Input Parameters

- **Academic Profile**: GPA, SAT/ACT scores, class rank
- **Preferences**: Location, major, campus size, cost range
- **Extracurriculars**: Activities, leadership roles, community service
- **Financial**: Budget constraints, need for financial aid

## 📈 Output

- Ranked list of suitable colleges
- Confidence scores for each recommendation
- Detailed college information
- Interactive visualizations
- Downloadable reports (PDF/CSV)

## 🔧 Configuration

The system can be customized by:
- Modifying the college dataset in `data/colleges.csv`
- Adjusting ML model parameters in `models/college_model.py`
- Customizing the UI in `frontend/streamlit_app.py`
- Adding new features to the API in `api/main.py`

## 🤝 Contributing

Feel free to contribute by:
- Adding new college data sources
- Improving the ML model
- Enhancing the UI/UX
- Adding new features
