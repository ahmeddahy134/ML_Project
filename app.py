import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc

# Page Configuration
st.set_page_config(
    page_title="Stroke Risk Prediction System",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        background-color: #f0f2f6;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        border-radius: 10px;
        padding: 10px 24px;
        font-size: 16px;
    }
    .stButton>button:hover {
        background-color: #45a049;
    }
    h1 {
        color: #2c3e50;
        text-align: center;
        padding: 20px;
    }
    h2 {
        color: #34495e;
    }
    .metric-card {
        background-color: white;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    </style>
    """, unsafe_allow_html=True)

# Main Title
st.title("🏥 Stroke Risk Prediction System")
st.markdown("---")

# Load Data
@st.cache_data
def load_data():
    """Load data from CSV file"""
    try:
        data = pd.read_csv("stroke_risk_dataset_v2.csv")
        # Drop duplicates as done in notebook
        data.drop_duplicates(inplace=True)
        return data
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

# Preprocess Data
@st.cache_data
def preprocess_data(data):
    data = data.drop(['excessive_sweating', 'nausea_vomiting'], axis=1)
    le = LabelEncoder()
    data['gender_encoded'] = le.fit_transform(data['gender'])
    feature_cols = ['age', 'gender_encoded', 'chest_pain', 'high_blood_pressure', 
                    'irregular_heartbeat', 'shortness_of_breath', 'fatigue_weakness',
                    'dizziness', 'swelling_edema', 'neck_jaw_pain',
                    'persistent_cough', 'chest_discomfort',
                    'cold_hands_feet', 'snoring_sleep_apnea', 'anxiety_doom']
    X = data[feature_cols]
    y = data['at_risk']
    return X, y, le, feature_cols

# Train Model
@st.cache_resource
def train_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    model = LogisticRegression()
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]

    accuracy = accuracy_score(y_test, y_pred)
    
    return model, scaler, accuracy, X_test, y_test, y_pred, y_pred_proba, X_train_scaled, X_test_scaled

data = load_data()

if data is not None:
    X, y, le, feature_cols = preprocess_data(data)
    model, scaler, accuracy, X_test, y_test, y_pred, y_pred_proba, X_train_scaled, X_test_scaled = train_model(X, y)
    
    st.sidebar.header("📊 Main Menu")
    page = st.sidebar.radio(
        "Choose Page:",
        ["Home", "Data Exploration", "Risk Prediction", "Model Evaluation"]
    )
    
    #  Home Page
    if page == "Home":
        st.header(" Home Page")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(" Total Records", f"{len(data):,}")
        
        with col2:
            st.metric(" At Risk", f"{data['at_risk'].sum():,}")
        
        with col3:
            st.metric(" Not At Risk", f"{(data['at_risk'] == 0).sum():,}")
        
        with col4:
            st.metric(" Model Accuracy", f"{accuracy*100:.2f}%")
        
        st.markdown("---")
        
        # Project Information
        st.subheader(" About the Project")
        st.write("""
        This system uses Machine Learning techniques to predict stroke risk based on various health and demographic factors.
        
        **Key Features:**
        -  Comprehensive health data analysis
        -  Logistic Regression Model (matching notebook)
        -  Interactive visualizations
        -  Accurate risk predictions
        
        **Model Details:**
        - Algorithm: Logistic Regression
        - Features Used: 15 features (matching notebook feature selection)
        - Test Size: 30%
        - Random State: 42
        
        **How to Use:**
        1. Explore data in the "Data Exploration" section
        2. Enter your data in the "Risk Prediction" section
        3. Review model performance in the "Model Evaluation" section
        """)
        
        # Sample Data
        st.subheader(" Sample Data")
        st.dataframe(data.head(10), use_container_width=True)
    
    #Data Exploration 
    elif page == "Data Exploration":
        st.header(" Data Exploration")
        tab1, tab2, tab3 = st.tabs(["Statistics", "Distributions", "Relationships"])
        
        with tab1:
            st.subheader(" Descriptive Statistics")
            st.dataframe(data.describe(), use_container_width=True)
            
            st.subheader(" Column Information")
            col_info = pd.DataFrame({
                'Data Type': data.dtypes,
                'Unique Values': data.nunique(),
                'Missing Values': data.isnull().sum()
            })
            st.dataframe(col_info, use_container_width=True)
            
            st.subheader("Features Used in Model")
            st.write(f"Total features: {len(feature_cols)}")
            for i, feature in enumerate(feature_cols, 1):
                st.write(f"{i}. {feature}")
        
        with tab2:
            st.subheader(" Variable Distributions")
            col1, col2 = st.columns(2)
            
            with col1:
                fig = px.pie(data, names='at_risk', 
                            title='Risk Status Distribution',
                            labels={'at_risk': 'Risk Status'},
                            color_discrete_sequence=['#2ecc71', '#e74c3c'])
                fig.update_traces(textposition='inside', textinfo='percent+label')
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                fig = px.histogram(data, x='age', nbins=30,
                                title='Age Distribution',
                                labels={'age': 'Age', 'count': 'Count'},
                                color_discrete_sequence=['#3498db'])
                st.plotly_chart(fig, use_container_width=True)
            
            # stroke_risk_percentage distribution
            fig = px.histogram(data, x='stroke_risk_percentage', nbins=50,
                            title='Stroke Risk Percentage Distribution',
                            labels={'stroke_risk_percentage': 'Risk Percentage (%)', 'count': 'Count'},
                            color_discrete_sequence=['#9b59b6'])
            st.plotly_chart(fig, use_container_width=True)
            
            # Gender distribution
            gender_counts = data['gender'].value_counts()
            fig = px.bar(x=gender_counts.index, y=gender_counts.values,
                        title='Gender Distribution',
                        labels={'x': 'Gender', 'y': 'Count'},
                        color=gender_counts.index,
                        color_discrete_sequence=['#3498db', '#e74c3c'])
            st.plotly_chart(fig, use_container_width=True)
        
        with tab3:
            st.subheader(" Variable Relationships")
            
            # Correlation matrix
            numeric_cols = data.select_dtypes(include=[np.number]).columns
            corr_matrix = data[numeric_cols].corr()
            
            fig = px.imshow(corr_matrix,
                        title='Correlation Matrix',
                        labels=dict(color="Correlation"),
                        color_continuous_scale='RdBu_r',
                        aspect='auto')
            st.plotly_chart(fig, use_container_width=True)
            
            # Age vs Risk Percentage
            fig = px.scatter(data, x='age', y='stroke_risk_percentage',
                        color='at_risk',
                        title='Age vs Risk Percentage',
                        labels={'age': 'Age', 'stroke_risk_percentage': 'Risk Percentage (%)', 'at_risk': 'At Risk'},
                        color_discrete_sequence=['#2ecc71', '#e74c3c'])
            st.plotly_chart(fig, use_container_width=True)
    
    #Risk Prediction 
    elif page == "Risk Prediction":
        st.header("🎯 Stroke Risk Prediction")
        
        st.write("Enter health data to get stroke risk prediction:")
        
        # Input Form 
        col1, col2, col3 = st.columns(3)
        
        with col1:
            age = st.number_input("Age", min_value=18, max_value=100, value=40)
            gender = st.selectbox("Gender", ["Male", "Female"])
            chest_pain = st.selectbox("Chest Pain", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
            high_blood_pressure = st.selectbox("High Blood Pressure", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
            irregular_heartbeat = st.selectbox("Irregular Heartbeat", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
        
        with col2:
            shortness_of_breath = st.selectbox("Shortness of Breath", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
            fatigue_weakness = st.selectbox("Fatigue/Weakness", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
            dizziness = st.selectbox("Dizziness", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
            swelling_edema = st.selectbox("Swelling/Edema", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
            neck_jaw_pain = st.selectbox("Neck/Jaw Pain", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
        
        with col3:
            persistent_cough = st.selectbox("Persistent Cough", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
            chest_discomfort = st.selectbox("Chest Discomfort", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
            cold_hands_feet = st.selectbox("Cold Hands/Feet", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
            snoring_sleep_apnea = st.selectbox("Snoring/Sleep Apnea", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
            anxiety_doom = st.selectbox("Anxiety", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
        
        # Predict Button
        if st.button(" Predict Risk", use_container_width=True):
            gender_encoded = le.transform([gender])[0]
            
            input_data = pd.DataFrame({
                'age': [age],
                'gender_encoded': [gender_encoded],
                'chest_pain': [chest_pain],
                'high_blood_pressure': [high_blood_pressure],
                'irregular_heartbeat': [irregular_heartbeat],
                'shortness_of_breath': [shortness_of_breath],
                'fatigue_weakness': [fatigue_weakness],
                'dizziness': [dizziness],
                'swelling_edema': [swelling_edema],
                'neck_jaw_pain': [neck_jaw_pain],
                'persistent_cough': [persistent_cough],
                'chest_discomfort': [chest_discomfort],
                'cold_hands_feet': [cold_hands_feet],
                'snoring_sleep_apnea': [snoring_sleep_apnea],
                'anxiety_doom': [anxiety_doom]
            })
            
            input_data = input_data[feature_cols]
            
            input_scaled = scaler.transform(input_data)
            
            # Predict
            prediction = model.predict(input_scaled)[0]
            prediction_proba = model.predict_proba(input_scaled)[0]
            
            # Display Results
            st.markdown("---")
            st.subheader(" Prediction Results")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if prediction == 1:
                    st.error("⚠️ **At Risk**")
                else:
                    st.success("✅ **Not At Risk**")
            
            with col2:
                st.metric("Risk Probability", f"{prediction_proba[1]*100:.2f}%")
            
            with col3:
                st.metric("Safety Probability", f"{prediction_proba[0]*100:.2f}%")
            
            # Probability Chart
            fig = go.Figure(data=[
                go.Bar(name='Not At Risk', x=['Result'], y=[prediction_proba[0]*100], marker_color='#2ecc71'),
                go.Bar(name='At Risk', x=['Result'], y=[prediction_proba[1]*100], marker_color='#e74c3c')
            ])
            fig.update_layout(
                title='Prediction Probabilities',
                yaxis_title='Probability (%)',
                barmode='group'
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Recommendations
            st.subheader("💡 Recommendations")
            if prediction == 1:
                st.warning("""
                **Strongly Recommended:**
                - 🏥 Consult a specialist doctor as soon as possible
                - 💊 Follow prescribed medications
                - 🏃‍♂️ Exercise regularly
                - 🥗 Follow a healthy diet
                - 🚭 Quit smoking
                - 😌 Reduce stress and anxiety
                """)
            else:
                st.info("""
                **To Maintain Your Health:**
                - ✅ Continue your healthy lifestyle
                - 🏃‍♂️ Exercise regularly
                - 🥗 Eat healthy and balanced meals
                - 🩺 Get regular check-ups
                - 😊 Maintain your mental health
                """)
    
    #Model Evaluation
    elif page == "Model Evaluation":
        st.header("📈 Model Performance Evaluation")
        
        tab1, tab2, tab3 = st.tabs(["Metrics", "Confusion Matrix", "ROC Curve"])
        
        with tab1:
            st.subheader("📊 Performance Metrics")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric(" Accuracy", f"{accuracy*100:.2f}%")
            
            # Calculate additional metrics
            from sklearn.metrics import precision_score, recall_score, f1_score
            
            precision = precision_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            
            with col2:
                st.metric("🎯 Precision", f"{precision*100:.2f}%")
            
            with col3:
                st.metric("🎯 Recall", f"{recall*100:.2f}%")
            
            st.metric("🎯 F1-Score", f"{f1*100:.2f}%")
            
            # Classification Report
            st.subheader("📋 Classification Report")
            report = classification_report(y_test, y_pred, output_dict=True)
            report_df = pd.DataFrame(report).transpose()
            st.dataframe(report_df, use_container_width=True)
            
            # Model Information
            st.subheader("🤖 Model Information")
            st.write(f"**Algorithm:** Logistic Regression")
            st.write(f"**Number of Features:** {len(feature_cols)}")
            st.write(f"**Test Set Size:** {len(y_test)} samples ({len(y_test)/len(data)*100:.1f}%)")
            st.write(f"**Training Set Size:** {len(data) - len(y_test)} samples")
            st.write(f"**Random State:** 42")
        
        with tab2:
            st.subheader("🔢 Confusion Matrix")
            
            cm = confusion_matrix(y_test, y_pred)
            
            fig = px.imshow(cm,
                        labels=dict(x="Predicted", y="Actual", color="Count"),
                        x=['Not At Risk', 'At Risk'],
                        y=['Not At Risk', 'At Risk'],
                        color_continuous_scale='Blues',
                        text_auto=True)
            fig.update_layout(title='Confusion Matrix')
            st.plotly_chart(fig, use_container_width=True)
            
            # Explanation
            st.info(f"""
            **Results Explanation:**
            -  **True Negatives (TN)**: {cm[0][0]} - Correct predictions of no risk
            -  **False Positives (FP)**: {cm[0][1]} - Incorrect predictions of risk
            -  **False Negatives (FN)**: {cm[1][0]} - Incorrect predictions of no risk
            -  **True Positives (TP)**: {cm[1][1]} - Correct predictions of risk
            """)
        
        with tab3:
            st.subheader("📈 ROC Curve")
            
            # Calculate ROC curve
            fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
            roc_auc = auc(fpr, tpr)
            
            # Plot ROC curve
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=fpr, y=tpr,
                                    mode='lines',
                                    name=f'ROC curve (AUC = {roc_auc:.2f})',
                                    line=dict(color='#3498db', width=2)))
            fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1],
                                    mode='lines',
                                    name='Random Classifier',
                                    line=dict(color='red', width=2, dash='dash')))
            fig.update_layout(
                title='ROC Curve (Receiver Operating Characteristic)',
                xaxis_title='False Positive Rate',
                yaxis_title='True Positive Rate',
                showlegend=True
            )
            st.plotly_chart(fig, use_container_width=True)
            
            st.success(f"**Area Under Curve (AUC)**: {roc_auc:.4f}")
            
            st.info("""
            **AUC Interpretation:**
            - 0.90 - 1.00: Excellent
            - 0.80 - 0.90: Very Good
            - 0.70 - 0.80: Good
            - 0.60 - 0.70: Fair
            - 0.50 - 0.60: Poor
            """)
            
            # Model Coefficients (Logistic Regression)
            st.subheader("🎯 Model Coefficients (Logistic Regression)")
            
            coefficients = pd.DataFrame({
                'Feature': feature_cols,
                'Coefficient': model.coef_[0]
            }).sort_values('Coefficient', ascending=False)
            
            fig = px.bar(coefficients, x='Coefficient', y='Feature',
                        orientation='h',
                        title='Feature Coefficients in Logistic Regression',
                        color='Coefficient',
                        color_continuous_scale='RdBu')
            fig.update_layout(xaxis_title='Coefficient Value')
            st.plotly_chart(fig, use_container_width=True)

else:
    st.error("❌ Failed to load data. Make sure stroke_risk_dataset_v2.csv file exists in the same folder.")
