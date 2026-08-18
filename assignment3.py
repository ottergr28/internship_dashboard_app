import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# streamlit page config
st.set_page_config(
    page_title="EDA Dashboard",  # the page title shown in the browser tab
    page_icon=":bar_chart:",  # the page favicon shown in the browser tab
    layout="wide",  # page layout : use the entire screen
)
# add page title
st.title("Full exploratory data analysis and an interactive dashboard based on anonymized patient health metrics.")

DATA_PATH = ("Dataset_Task_3_Health_Data.csv")
@st.cache_data(ttl=3600)
def load_clean_data(path):
    df = pd.read_csv(path, sep=';',decimal=',')
    df.drop(df.columns[df.columns.str.contains(
    'unnamed', case=False)], axis=1, inplace=True)
    df = df.dropna()
    return df

df = load_clean_data(DATA_PATH)

attributes = df.columns[:-1].str.replace("_"," ").tolist()
st.header("Basic Statistics")
st.dataframe(df.describe(), width='stretch')

#sidebar with filtering 
with st.sidebar:
    st.header("Filters")
    # Age range from the dataframe
    age_min = int(df['Age'].min())
    age_max = int(df['Age'].max())
    age_range = st.slider("Age range", min_value=age_min, max_value=age_max, value=(age_min, age_max))

    # Disease type multiselect 
    disease_options = sorted(df['Disease_Status'].unique().tolist())
    selected_diseases = st.multiselect("Disease_Status (select one or more)", options=disease_options,
                                       default=disease_options)

    
    activity_options = sorted(df['Activity_Level'].unique().tolist())
    selected_activity = st.multiselect("Activity_Level", options=activity_options, default=activity_options)

# Apply filters to produce df_filtered (use inclusive defaults if user clears selection)
df_filtered = df.copy()
# Age filter
df_filtered = df_filtered[df_filtered['Age'].between(age_range[0], age_range[1])]

# Disease_Status filter (if the user selected none, keep all - change behavior if you prefer)
if selected_diseases:
    df_filtered = df_filtered[df_filtered['Disease_Status'].isin(selected_diseases)]

# plotly graphs:
st.header("Data Visualization")

numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
categorical_columns = df.select_dtypes(include=['str']).columns.tolist()

# # Converting categorical variables into numerical format for correlation analysis
# df_encoded = df_filtered.copy()
# df_encoded.drop(columns=["Patient_ID"], inplace=True)

from sklearn.preprocessing import OrdinalEncoder

# Now create an encoded copy from the filtered df for plots that need encoded numeric columns
def encode_df_for_plots(df_in):
    df_encoded = df_in.copy()
    if 'Patient_ID' in df_encoded.columns:
        df_encoded = df_encoded.drop(columns=['Patient_ID'])
    # map activity level (if it's ordinal Low/Moderate/High)
    if 'Activity_Level' in df_encoded.columns:
        encoder = OrdinalEncoder(categories=[['Low', 'Moderate', 'High']])
        df_encoded['Activity_Level'] = encoder.fit_transform(df_filtered[['Activity_Level']])
    # convert Disease_Status to category codes so heatmaps / correlations can use it
    if 'Disease_Status' in df_encoded.columns:
        df_encoded['Disease_Status'] = df_encoded['Disease_Status'].astype('category').cat.codes
    return df_encoded

df_encoded = encode_df_for_plots(df_filtered)
# encoder = OrdinalEncoder(categories=[['Low', 'Moderate', 'High']])
# df_encoded['Activity_Level'] = encoder.fit_transform(df_filtered[['Activity_Level']])
# df_encoded['Disease_Status'] = df_encoded['Disease_Status'].astype('category').cat.codes

#counter histograms
def histplot(df):
    fig, axs = plt.subplots(2, 2, figsize=(15, 15))

    for i, col in enumerate(numeric_columns):
        row = i // 2
        col = i % 2
        sns.histplot(df[numeric_columns[i]], ax=axs[row, col])

    plt.title('Counter Histogram')

    return fig

#correlations
def plot_correlation(df):
    # Calculating the correlations
    correlations = df.corr()

    # Plotting the correlations in a heatmap
    fig = plt.figure(figsize=(15,10))
    sns.heatmap(correlations, annot=True, fmt=".2f", cmap='coolwarm', cbar=True)
    plt.title('Correlations')
    return fig

def crosstab_activity_disease(df):
    fig = plt.figure(figsize=(15,10))
    ct = pd.crosstab(df['Activity_Level'], df['Disease_Status'])
    sns.heatmap(ct, annot=True, fmt='d', cmap='YlGnBu')
    plt.title('Disease Status by Activity Level')

    return fig

def swarmplot_disease_bmi(df):
    fig = plt.figure(figsize=(8,5))
    sns.violinplot(x='Disease_Status', y='BMI', data=df, inner=None, color=".8")
    sns.swarmplot(x='Disease_Status', y='BMI', data=df, size=3, color='k', alpha=0.6)
    plt.title('Disease Status by BMI')

    return fig

def scatterplot_bmi_disease_age(df):
    x_j = df['Disease_Status'].astype(float) + np.random.normal(0, 0.08, size=len(df_encoded))

    fig = plt.figure(figsize=(8,5))
    sc = plt.scatter(x_j, df['BMI'], c=df['Age'], cmap='viridis', alpha=0.6, s=30)
    plt.xticks(sorted(df['Disease_Status'].unique()))
    plt.xlabel('Disease_Status'); plt.ylabel('BMI')
    cbar = plt.colorbar(sc)
    cbar.set_label('Age')
    plt.title('Disease Status by BMI')

    return fig

def scatterplot_cholesterol_disease(df):
    x_j = df['Disease_Status'].astype(float) + np.random.normal(0, 0.08, size=len(df_encoded))

    fig = plt.figure(figsize=(8,5))
    sc = plt.scatter(x_j, df['Cholesterol_mg_dL'], c=df['Age'], cmap='viridis', alpha=0.6, s=30)
    plt.xticks(sorted(df['Disease_Status'].unique()))
    plt.xlabel('Disease_Status'); plt.ylabel('Cholesterol_mg_dL')
    cbar = plt.colorbar(sc)
    cbar.set_label('Age')
    plt.title('Disease Status by Cholesterol')

    return fig

def scatterplot_cholesterol_bmi(df):
    fig = plt.figure(figsize=(7,5))
    sns.scatterplot(x='BMI', y='Cholesterol_mg_dL', hue='Disease_Status',palette=sns.color_palette('viridis'), data=df, alpha=0.6)
    sns.regplot(x='BMI', y='Cholesterol_mg_dL', data=df, scatter=False, color='black')
    plt.title('BMI by Cholesterol')

    return fig

def pairplot(df):
    cols = ['BMI','Cholesterol_mg_dL','Glucose_mg_dL','Disease_Status']
    df2 = df[cols].dropna()
    g = sns.pairplot(
        df2,
        hue='Disease_Status',
        diag_kind='kde',
        plot_kws={'alpha': 0.6, 's': 20},
        palette=sns.color_palette('viridis'),
        corner=False   # set True to show only lower triangle
    )

    plt.title('Pairplot')

    return g.figure


fig = histplot(df_filtered)
st.pyplot(fig)
plt.close(fig)

fig = plot_correlation(df_encoded)
st.pyplot(fig)
plt.close(fig)

fig = crosstab_activity_disease(df_filtered)
st.pyplot(fig)
plt.close(fig)

fig = swarmplot_disease_bmi(df_encoded)
st.pyplot(fig)
plt.close(fig)

fig = scatterplot_bmi_disease_age(df_encoded)
st.pyplot(fig)
plt.close(fig)

fig = scatterplot_cholesterol_disease(df_encoded)
st.pyplot(fig)
plt.close(fig)

fig = scatterplot_cholesterol_bmi(df_encoded)
st.pyplot(fig)
plt.close(fig)

fig = pairplot(df_encoded)
st.pyplot(fig)
plt.close(fig)


#summary
grouped = df_filtered.groupby('Disease_Status').agg(
    n=('BMI','size'),
    mean_BMI=('BMI','mean'),
    sd_BMI=('BMI','std'),
    mean_Chol=('Cholesterol_mg_dL','mean'),
    sd_Chol=('Cholesterol_mg_dL','std'),
    mean_Glu=('Glucose_mg_dL','mean'),
    sd_Glu=('Glucose_mg_dL','std'),
).reset_index()
st.write(grouped)

#correlations
num = df_filtered[['BMI','Cholesterol_mg_dL','Glucose_mg_dL','Age']].dropna()
st.write("Pearson:\n", num.corr(method='pearson').round(3))
st.write("Spearman:\n", num.corr(method='spearman').round(3))
