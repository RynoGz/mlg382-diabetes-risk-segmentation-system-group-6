import dash
from dash import html, dcc, Input, Output, State
import pandas as pd
import numpy as np
import joblib
import os

# 1. Path Setup
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, 'artifacts', 'classification_model.pkl')
SCALER_PATH = os.path.join(BASE_DIR, 'artifacts', 'scaler.pkl')

# 2. Initialize App
app = dash.Dash(__name__)
server = app.server

# 3. Load Group Artifacts
model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

# 4. Feature Lists
CONTINUOUS_FEATURES = scaler.feature_names_in_.tolist()
ALL_MODEL_FEATURES = [
    'Age_std', 'alcohol_consumption_per_week_std', 'physical_activity_minutes_per_week_std',
    'diet_score_std', 'sleep_hours_per_day_std', 'screen_time_hours_per_day_std', 'bmi_std',
    'waist_to_hip_ratio_std', 'systolic_bp_std', 'diastolic_bp_std', 'heart_rate_std',
    'cholesterol_total_std', 'hdl_cholesterol_std', 'ldl_cholesterol_std', 'triglycerides_std',
    'glucose_fasting_std', 'glucose_postprandial_std', 'insulin_level_std', 'hba1c_std',
    'diabetes_risk_score_std', 'pulse_pressure_std', 'cholesterol_ratio_std', 'ldl_hdl_ratio_std',
    'homa_ir_std', 'glucose_spike_std', 'education_level_enc', 'income_level_enc',
    'family_history_diabetes', 'hypertension_history', 'cardiovascular_history',
    'gender_Male', 'gender_Other', 'ethnicity_Black', 'ethnicity_Hispanic',
    'ethnicity_Other', 'ethnicity_White', 'employment_status_Retired',
    'employment_status_Student', 'employment_status_Unemployed', 'smoking_status_Former',
    'smoking_status_Never', 'bmi_category', 'glucose_fasting_category',
    'hba1c_category', 'bp_category', 'low_activity_flag', 'poor_diet_flag',
    'high_alcohol_flag', 'poor_sleep_flag', 'high_screen_flag', 'lifestyle_risk_score',
    'age_group', 'metabolic_syndrome_score', 'metabolic_syndrome_flag',
    'comorbidity_count', 'high_triglycerides', 'low_hdl', 'high_glucose',
    'high_bp_flag', 'high_whr'
]

# 5. Layout
app.layout = html.Div([
    html.Div([
        html.H1("BC Analytics: Diabetes Decision Support", style={'color': '#2c3e50', 'margin': '0'}),
        html.P("Validated Clinical Segmentation Engine", style={'color': '#7f8c8d'})
    ], style={'textAlign': 'center', 'padding': '30px', 'backgroundColor': 'white', 'borderBottom': '1px solid #ddd'}),

    html.Div([
        html.Div([
            html.H4("Patient Consultation Inputs", style={'marginBottom': '20px'}),
            html.Div([
                # Left Column
                html.Div([
                    html.Label("Age:"), dcc.Input(id='age-in', type='number', value=30, style={'width': '100%', 'marginBottom': '10px'}),
                    html.Label("BMI:"), dcc.Input(id='bmi-in', type='number', value=22.0, step=0.1, style={'width': '100%', 'marginBottom': '10px'}),
                    html.Label("Exercise (min/wk):"), dcc.Input(id='act-in', type='number', value=150, style={'width': '100%', 'marginBottom': '10px'}),
                    html.Label("Gender:"), dcc.Dropdown(id='gender-in', options=[{'label': 'Female', 'value': 'F'}, {'label': 'Male', 'value': 'M'}], value='M'),
                ], style={'width': '45%', 'display': 'inline-block'}),
                
                # Right Column
                html.Div([
                    html.Label("HbA1c (%):"), dcc.Input(id='hba1c-in', type='number', value=5.2, step=0.1, style={'width': '100%', 'marginBottom': '10px'}),
                    html.Label("Fasting Glucose:"), dcc.Input(id='gluc-in', type='number', value=90, style={'width': '100%', 'marginBottom': '10px'}),
                    html.Label("Systolic BP:"), dcc.Input(id='bp-in', type='number', value=118, style={'width': '100%', 'marginBottom': '10px'}),
                    html.Label("Family History:"), dcc.Dropdown(id='fam-in', options=[{'label': 'No', 'value': 0}, {'label': 'Yes', 'value': 1}], value=0),
                ], style={'width': '45%', 'display': 'inline-block', 'float': 'right'}),
            ], style={'marginBottom': '20px'}),
            
            html.Button('GENERATE ASSESSMENT', id='calc-btn', n_clicks=0, 
                        style={'width': '100%', 'padding': '15px', 'backgroundColor': '#27ae60', 'color': 'white', 'border': 'none', 'borderRadius': '5px', 'fontSize': '16px', 'cursor': 'pointer'})
        ], style={'maxWidth': '800px', 'margin': '0 auto', 'backgroundColor': 'white', 'padding': '40px', 'borderRadius': '10px', 'boxShadow': '0 4px 15px rgba(0,0,0,0.1)'}),

        html.Div(id='result-display', style={'marginTop': '30px', 'textAlign': 'center'})
    ], style={'padding': '40px', 'backgroundColor': '#f9f9f9', 'minHeight': '100vh'})
])

# 6. Prediction Logic
@app.callback(
    Output('result-display', 'children'),
    Input('calc-btn', 'n_clicks'),
    [State('age-in', 'value'), State('bmi-in', 'value'), State('act-in', 'value'),
     State('gender-in', 'value'), State('hba1c-in', 'value'), State('gluc-in', 'value'),
     State('bp-in', 'value'), State('fam-in', 'value')]
)
def run_model(n_clicks, age, bmi, act, gender, hba1c, gluc, bp, fam):
    if n_clicks > 0:
        try:
            # A. HEALTHY BASELINE INITIALIZATION
            # We set background values (Insulin, Diet Score, etc.) to HEALTHY constants
            # instead of using the diabetic mean of the group's dataset.
            healthy_defaults = {
                'insulin_level': 7.0, 'glucose_postprandial': 110, 'glucose_spike': 30,
                'triglycerides': 100, 'diet_score': 8.0, 'waist_to_hip_ratio': 0.82,
                'cholesterol_total': 170, 'hdl_cholesterol': 60, 'ldl_cholesterol': 90
            }
            
            input_df = pd.DataFrame([scaler.mean_], columns=CONTINUOUS_FEATURES)
            for k, v in healthy_defaults.items():
                input_df[k] = v
            
            # Inject User Inputs
            input_df['Age'] = age
            input_df['bmi'] = bmi
            input_df['physical_activity_minutes_per_week'] = act
            input_df['hba1c'] = hba1c
            input_df['glucose_fasting'] = gluc
            input_df['systolic_bp'] = bp
            
            scaled_values = scaler.transform(input_df)
            model_input = pd.DataFrame(np.zeros((1, 60)), columns=ALL_MODEL_FEATURES)
            model_input.iloc[0, 0:25] = scaled_values[0]
            
            # B. SYNC CATEGORICAL FLAGS
            # (Matches raw numbers to the boolean flags inside the Random Forest)
            model_input['gender_Male'] = 1 if gender == 'M' else 0
            model_input['family_history_diabetes'] = fam
            model_input['bmi_category'] = 3 if bmi >= 30 else 2 if bmi >= 25 else 1
            model_input['age_group'] = 3 if age >= 60 else 2 if age >= 45 else 1
            model_input['hba1c_category'] = 2 if hba1c >= 6.5 else 1 if hba1c >= 5.7 else 0
            model_input['glucose_fasting_category'] = 2 if gluc >= 126 else 1 if gluc >= 100 else 0
            
            # C. PREDICT (Alphabetical: 0:G, 1:No, 2:Pre, 3:T1, 4:T2)
            pred_idx = model.predict(model_input)[0]
            label_map = {0: "Gestational", 1: "No Diabetes (Healthy)", 2: "Pre-diabetes Risk", 3: "Type 1", 4: "Type 2"}
            label = label_map.get(pred_idx, f"Stage {pred_idx}")
            
            color = "#27ae60" if pred_idx == 1 else "#e67e22" if pred_idx == 2 else "#e74c3c"
            return html.Div([
                html.H2(f"Assessment: {label}", style={'color': color, 'margin': '0'}),
                html.P("Clinical logic applied successfully.")
            ], style={'padding': '20px', 'backgroundColor': 'white', 'borderRadius': '10px', 'borderLeft': f'10px solid {color}', 'maxWidth': '600px', 'margin': '0 auto'})
            
        except Exception as e:
            return html.Div(f"System Error: {str(e)}", style={'color': 'red'})
    return ""

if __name__ == '__main__':
    app.run(debug=True)