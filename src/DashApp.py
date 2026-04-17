import dash
from dash import html, dcc, Input, Output, State
import pandas as pd
import joblib
import os
import numpy as np

# 1. Setup paths to find your team's "brains"
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, 'artifacts', 'classification_model.pkl')
SCALER_PATH = os.path.join(BASE_DIR, 'artifacts', 'scaler.pkl')

# 2. Initialize the Dash app
app = dash.Dash(__name__)
server = app.server

# 3. App Layout
app.layout = html.Div([
    html.H1("BC Analytics: Diabetes Risk Decision Support"),
    html.Div([
        html.H3("Patient Data Input"),
        html.Label("Age:"), dcc.Input(id='age', type='number', value=30),
        html.Br(),
        html.Label("BMI:"), dcc.Input(id='bmi', type='number', value=25),
        html.Br(),
        html.Button('Calculate Risk Assessment', id='predict-btn', n_clicks=0),
    ], style={'padding': '20px', 'backgroundColor': '#f9f9f9'}),
    
    html.Hr(),
    html.Div(id='prediction-output', style={'fontSize': '20px', 'fontWeight': 'bold'})
])

# 4. Prediction Logic
@app.callback(
    Output('prediction-output', 'children'),
    Input('predict-btn', 'n_clicks'),
    [State('age', 'value'), State('bmi', 'value')]
)
def predict_risk(n_clicks, age, bmi):
    if n_clicks > 0:
        if os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH):
            # We will add the scaling and prediction logic here 
            # once your .pkl files are generated!
            return f"Assessment Successful for Patient (Age: {age})"
        else:
            return "Error: Model files not found in artifacts folder. Please run training scripts."
    return "Enter patient data and click Calculate."

if __name__ == '__main__':
    app.run(debug=True)