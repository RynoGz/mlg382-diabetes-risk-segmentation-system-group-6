#Import the libraries we need
import pandas as pd                    
import seaborn as sns                  
import matplotlib.pyplot as plt        
from sklearn.preprocessing import StandardScaler   
from sklearn.cluster import KMeans                
from sklearn.decomposition import PCA              

#Load the dataset from the 'data' folder
df = pd.read_csv('../data/Diabetes_and_LifeStyle_Dataset_.csv')

#Print the first few rows to make sure it loaded correctly
print("First 5 rows of the dataset:")
print(df.head())

#Check how many rows and columns we have
print("\nDataset shape:", df.shape)

#Choose the lifestyle features for clustering
#These are habits that people can change to reduce diabetes risk
features = [
    'physical_activity_minutes_per_week',   
    'diet_score',                           
    'sleep_hours_per_day',                  
    'bmi',                                  
    'alcohol_consumption_per_week'          
]
#Show the first few rows of just these features
print("\nLifestyle features (first 5 rows):")
print(df[features].head())

#Scale the features (very important for K-Means)
#Different features have different units (minutes, scores, kg/m2, etc.)
#Scaling puts them all on the same scale (mean=0, standard deviation=1)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df[features])

print("\nAfter scaling, the data looks like this (first 5 rows):")
print(X_scaled[:5])

#Run K-Means clustering with k = 3 
#We use random_state=42 so that results are the same every time we run it
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)

#Fit the model and predict which cluster each person belongs to
df['Lifestyle_Segment'] = kmeans.fit_predict(X_scaled)

#Count how many people are in each cluster
print("\nNumber of people in each cluster (0,1,2):")
print(df['Lifestyle_Segment'].value_counts())

#Give each cluster a meaningful name (Persona)
#Based on looking at the cluster characteristics (we did this analysis earlier)
#Cluster 0: High activity = Active Movers
#Cluster 1: Low activity + high BMI = High-Risk Inactive
#Cluster 2: High diet score = Mindful Eaters
persona_map = {
    0: 'The Active Movers',
    1: 'The High-Risk Inactive',
    2: 'The Mindful Eaters'
}
df['Persona'] = df['Lifestyle_Segment'].map(persona_map)

#Show the first few rows with the new Persona column
print("\nFirst 10 rows with Persona assigned:")
print(df[['Lifestyle_Segment', 'Persona'] + features].head(10))

#Reduce the data to 2 dimensions so we can plot it (PCA)
pca = PCA(n_components=2)
pca_results = pca.fit_transform(X_scaled)

#Add the two new columns to our dataframe
df['PC1'] = pca_results[:, 0]   
df['PC2'] = pca_results[:, 1]   

print("\nHow much variance is explained by PC1 and PC2?")
print("PC1 explains:", round(pca.explained_variance_ratio_[0] * 100, 2), "%")
print("PC2 explains:", round(pca.explained_variance_ratio_[1] * 100, 2), "%")

#Make a comparison graph (2 side-by-side plots)
fig, axes = plt.subplots(1, 2, figsize=(22, 10))

#Colors for the personas
persona_colors = {
    'The Active Movers': 'green',
    'The High-Risk Inactive': 'red',
    'The Mindful Eaters': 'blue'
}

#Colors for the actual diabetes stages
disease_colors = {
    'No Diabetes': 'green',
    'Pre-Diabetes': 'orange',
    'Type 2': 'red'
}

#GRAPH A: Our lifestyle personas (based on habits)
sns.scatterplot(
    data=df,
    x='PC1',
    y='PC2',
    hue='Persona',
    palette=persona_colors,
    alpha=0.4,     
    s=30,           
    ax=axes[0]
)
axes[0].set_title("1. LIFESTYLE PERSONAS\n(Groups defined by Habits)", fontsize=15, fontweight='bold')
axes[0].legend(title="Persona")

#GRAPH B: actual diabetes diagnosis
#Only keep the three main stages (ignore any rare values)
target_stages = ['No Diabetes', 'Pre-Diabetes', 'Type 2']
df_filtered = df[df['diabetes_stage'].isin(target_stages)]

sns.scatterplot(
    data=df_filtered,
    x='PC1',
    y='PC2',
    hue='diabetes_stage',
    palette=disease_colors,
    alpha=0.4,
    s=30,
    ax=axes[1]
)
axes[1].set_title("2. CLINICAL REALITY\n(Actual Diabetes Diagnosis)", fontsize=15, fontweight='bold')
axes[1].legend(title="Diabetes Stage")

# Adjust the layout so the plots don't overlap
plt.tight_layout()

# Show the graphs
plt.show()

#Calculate the capture rate (risk analysis)
#This table shows, for each Persona, what percentage has No Diabetes, Pre-Diabetes, or Type 2
#We use normalize='index' to get percentages within each row (persona)
#RISK ANALYSIS

#ROW PERCENTAGES (what % of each Persona has each stage)
#Persona
row_table = pd.crosstab(df['Persona'], df['diabetes_stage'], normalize='index') * 100

print("\n" + "="*60)
print("VIEW 1: WITHIN EACH PERSONA – what % has each diabetes stage?")
print("(Each row sums to 100%)")
print("="*60)
print(row_table[['No Diabetes', 'Pre-Diabetes', 'Type 2']].round(2))

#COLUMN PERCENTAGES (what % of each diabetes stage falls into each Persona)
#Diabetes stage
col_table = pd.crosstab(df['Persona'], df['diabetes_stage'], normalize='columns') * 100

print("\n" + "="*60)
print("VIEW 2: TOTAL POPULATION SPREAD – for each diabetes stage, what % are in each Persona?")
print("(Each column sums to 100%)")
print("="*60)
print(col_table[['No Diabetes', 'Pre-Diabetes', 'Type 2']].round(2))

#Highlight the Type 2 column as the most important
print("\n🔴 KEY INSIGHT: Distribution of Type 2 patients across personas:")
type2_distribution = col_table['Type 2'].sort_values(ascending=False)
for persona, pct in type2_distribution.items():
    print(f"   {persona}: {pct:.1f}%")

#Print a summary of the cluster characteristics (average values)
print("\n" + "="*60)
print("CLUSTER PROFILES (average lifestyle values per persona)")
print("="*60)
print(df.groupby('Persona')[features].mean().round(1))

#Save the results (optional)
#We can save the dataframe with the new Persona column to a CSV file
df.to_csv('segmented_patients.csv', index=False)
print("\nSaved the segmented dataset to 'segmented_patients.csv'")
