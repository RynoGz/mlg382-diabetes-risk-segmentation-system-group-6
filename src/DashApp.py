import dash
from dash import html, dcc
import pandas as pd

# 1. Initialize the App
app = dash.Dash(__name__)
server = app.server # Required for Render deployment

# 2. The Layout 
app.layout = html.Div([
    html.H1("BC Analytics: Diabetes Risk Decision Support"),
    html.P("Enter patient lifestyle data below to determine risk stage."),
    
    # Placeholder Inputs 
    html.Div([
        html.Label("Age: "),
        dcc.Input(id='input-age', type='number', value=30),
        html.Br(),
        html.Label("BMI: "),
        dcc.Input(id='input-bmi', type='number', value=25),
    ]),
    
    html.Button('Calculate Risk & Segment', id='predict-button', n_clicks=0),
    
    html.Hr(),
    html.H3("Results"),
    html.Div(id='risk-output'), # Where the Classification result goes
    html.Div(id='segment-output'), # Where the K-Means result goes
    html.Div(id='driver-analysis') # Where the SHAP values go
])

if __name__ == '__main__':
    app.run(debug=True)