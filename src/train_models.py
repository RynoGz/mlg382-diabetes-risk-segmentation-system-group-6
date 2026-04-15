import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

# Look for the file in the 'data' folder that is one level up from your 'notebooks' folder
df = pd.read_csv('../data/Diabetes_and_LifeStyle_Dataset_.csv')

# Check if it loaded successfully
df.head()

# --- 1. DATA PREPARATION ---
# Selecting features that define a "Lifestyle"
features = ['physical_activity_minutes_per_week', 'diet_score', 'sleep_hours_per_day', 'bmi', 'alcohol_consumption_per_week']
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df[features])

# --- 2. CLUSTERING & PERSONA NAMING ---
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
df['Lifestyle_Segment'] = kmeans.fit_predict(X_scaled)

# Mapping generic numbers to Descriptive Personas based on our previous analysis
# Segment 0: High Activity, Segment 1: Low Activity/High BMI, Segment 2: High Diet
persona_map = {
    0: 'The Active Movers',
    1: 'The High-Risk Sedentary',
    2: 'The Mindful Eaters'
}
df['Persona'] = df['Lifestyle_Segment'].map(persona_map)

# --- 3. DIMENSIONALITY REDUCTION (For the 2D Map) ---
pca = PCA(n_components=2)
pca_results = pca.fit_transform(X_scaled)
df['PC1'] = pca_results[:, 0]
df['PC2'] = pca_results[:, 1]

# --- 4. THE COMPARISON VISUALIZATION ---
fig, axes = plt.subplots(1, 2, figsize=(22, 10))

# Custom Colors for Medical Clarity
persona_colors = {'The Active Movers': 'green', 'The High-Risk Sedentary': 'red', 'The Mindful Eaters': 'blue'}
disease_colors = {'No Diabetes': 'green', 'Pre-Diabetes': 'orange', 'Type 2': 'red'}

# GRAPH A: Our Designed Personas
sns.scatterplot(data=df, x='PC1', y='PC2', hue='Persona', 
                palette=persona_colors, alpha=0.4, s=30, ax=axes[0])
axes[0].set_title("1. LIFESTYLE PERSONAS\n(Groups defined by Habits)", fontsize=15, fontweight='bold')

# GRAPH B: The Clinical Reality (Filtered for focus)
target_stages = ['No Diabetes', 'Pre-Diabetes', 'Type 2']
df_filtered = df[df['diabetes_stage'].isin(target_stages)]

sns.scatterplot(data=df_filtered, x='PC1', y='PC2', hue='diabetes_stage', 
                palette=disease_colors, alpha=0.4, s=30, ax=axes[1])
axes[1].set_title("2. CLINICAL REALITY\n(Actual Diabetes Diagnosis)", fontsize=15, fontweight='bold')

# Clean up layout
plt.tight_layout()
plt.show()

# --- 5. THE DECISION TABLE (The "Truth") ---
# This proves which Persona is actually the "Diabetes Hotspot"
truth_table = pd.crosstab(df['Persona'], df['diabetes_stage'], normalize='index') * 100
print("\n" + "="*50)
print("RISK ANALYSIS: What % of each Persona has which condition?")
print("="*50)
print(truth_table[['No Diabetes', 'Pre-Diabetes', 'Type 2']].round(2))