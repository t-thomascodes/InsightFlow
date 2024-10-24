import pandas as pd
import streamlit as st
import io
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.preprocessing import StandardScaler

# Set page config for Solarized Dark theme
st.set_page_config(page_title="InsightFlow", layout="wide", initial_sidebar_state="expanded")

# Updated Custom CSS
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@300;400;700&display=swap');

    /* Solarized Dark color palette */
    :root {
        --base03:  #002b36;
        --base02:  #073642;
        --base01:  #586e75;
        --base00:  #657b83;
        --base0:   #839496;
        --base1:   #93a1a1;
        --base2:   #eee8d5;
        --base3:   #fdf6e3;
        --yellow:  #b58900;
        --orange:  #cb4b16;
        --red:     #dc322f;
        --magenta: #d33682;
        --violet:  #6c71c4;
        --blue:    #808080;
        --cyan:    #2aa198;
        --green:   #859900;
    }

    body {
        font-family: 'Roboto', sans-serif;
        background-color: var(--base03);
        color: var(--base0);
        line-height: 1.6;
    }

    h1 {
        font-size: 36px;
        font-weight: 700;
        margin-bottom: 30px;
        border-bottom: 1px solid var(--base01);
        padding-bottom: 10px;
    }

    h2, h3 {
        font-size: 24px;
        font-weight: 700;
        margin-top: 30px;
        margin-bottom: 20px;
        border-bottom: 1px solid var(--base01);
        padding-bottom: 10px;
    }

    p, li {
        font-size: 18px;
    }

    .stButton>button {
        background-color: var(--blue);
        color: var(--base3);
        border: none;
        border-radius: 6px;
        padding: 10px 20px;
        font-size: 18px;
        font-weight: 700;
        box-shadow: 2px 2px 5px rgba(0, 0, 0, 0.2);
        transition: background-color 0.3s ease;
    }

    .stButton>button:hover {
        background-color: var(--cyan);
    }

    .stDataFrame {
        background-color: var(--base02);
        border: 1px solid var(--base01);
        margin-bottom: 20px;
    }

    .stDataFrame th {
        background-color: var(--base01);
        color: var(--base3);
        font-weight: 700;
        background-image: linear-gradient(to bottom, var(--base01), var(--base02));
    }

    .stDataFrame td {
        background-color: var(--base03);
    }

    .stDataFrame tr:nth-child(even) td {
        background-color: var(--base02);
    }

    .stDataFrame th, .stDataFrame td {
        border: 1px solid var(--base01);
        padding: 8px;
    }

    .upload-section {
        border-bottom: 1px solid var(--base01);
        padding: 20px 0;
        margin-bottom: 30px;
        text-align: center;
    }

    .info-section {
        border-bottom: 1px solid var(--base01);
        padding-bottom: 20px;
        margin-bottom: 30px;
    }

    .info-section p {
        margin-bottom: 10px;
    }

    .info-section strong {
        color: var(--yellow);
    }

    .stAlert {
        background-color: var(--base02);
        color: var(--base1);
        padding: 15px;
        border-radius: 6px;
        margin-top: 20px;
        border: 1px solid var(--base01);
    }
</style>
""", unsafe_allow_html=True)

# Helper functions (same as before)
def load_dataset(file):
    return pd.read_csv(file)

def display_basic_info(df):
    st.markdown("<h2>Original Dataset Overview</h2>", unsafe_allow_html=True)
    st.dataframe(df.head(), use_container_width=True)
    st.markdown(f"<p><strong>Total number of elements:</strong> {df.size}</p>", unsafe_allow_html=True)
    st.markdown(f"<p><strong>Number of columns:</strong> {len(df.columns)}</p>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

def display_dataset_analysis(df):
    buffer = io.StringIO()
    df.info(buf=buffer)
    info_data = buffer.getvalue()

    st.text(info_data)
    st.write(df.describe())

    null_values = df.isnull().sum().sum()
    duplicated_values = df.duplicated().sum()
    non_unique_count = 0

    for col in df.select_dtypes(include=['object', 'category']).columns:
        if df[col].nunique() != df[col].shape[0]:
            non_unique_count += 1

    st.markdown("<div class='info-section'>", unsafe_allow_html=True)
    st.markdown(f"<p><strong>Null values:</strong> {null_values}</p>", unsafe_allow_html=True)
    st.markdown(f"<p><strong>Duplicated values:</strong> {duplicated_values}</p>", unsafe_allow_html=True)
    st.markdown(f"<p><strong>Non-unique categorical columns:</strong> {non_unique_count}</p>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)



def clean_dataset_null(df):
    null_values = df.isnull().sum().sum()

    if null_values > 0:
        clean_option = st.selectbox(
            "How would you like to handle the missing values?",
            ('Do nothing', 'Fill with Mean', 'Fill with Median', 'Drop Rows', 'Drop Columns')
        )
        
        st.session_state['clean_option'] = clean_option

        if clean_option == 'Fill with Mean':
            df = df.fillna(df.mean())
            st.success("Filled null values with the column mean.")
        elif clean_option == 'Fill with Median':
            df = df.fillna(df.median())
            st.success("Filled null values with the column median.")
        elif clean_option == 'Drop Rows':
            df = df.dropna()
            st.success("Dropped rows containing null values.")
        elif clean_option == 'Drop Columns':
            df = df.dropna(axis=1)
            st.success("Dropped columns containing null values.")
        elif clean_option == 'Do nothing':
            st.info("No changes made to the dataset.")
    else:
        st.success("No missing values found in the dataset.")
    return df

def clean_dataset_duplicated(df):
    duplicated_values = df.duplicated().sum()
    st.write(f'The dataset has {duplicated_values} duplicated values')

    if duplicated_values > 0:
        if st.button('Remove duplicated values'):
            df = df.drop_duplicates()
            st.write("Removed duplicated rows.")
    else:
        st.write('No duplicate values in dataset')
    
    return df

def clean_dataset_formatting(df):
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = df[col].str.strip()
    st.write("Removed trailing spaces from string columns.")
    return df

def display_errors(df):

    null_values = df.isnull().sum().sum()
    duplicated_values = df.duplicated().sum()
    non_unique_count = 0

    for col in df.select_dtypes(include=['object', 'category']).columns:
        if df[col].nunique() != df[col].shape[0]:
            non_unique_count += 1

    st.write(f'Number of null values: {null_values}')
    st.write(f'Number of duplicated values: {duplicated_values}')
    st.write(f'NUmber of non unique catergorical variables: {non_unique_count}')


def chooseOption():
#Graph options
        option = st.selectbox(
            "What type of graph do you want to use to analyze your data?",
            ("Bar Graph", "Scatter Plot", "Line Plot", "Pie Chart")
        )
        st.write(f"You selected: {option}")
        return option

def chooseXVariable(df):

    columns = df.columns.tolist()

    xVariable = st.selectbox(
        "What variable do you want to plot as your independent variable",
        (columns)
    )
    st.write(f"You chose: {xVariable} as your independent variable")
    
    return xVariable

def chooseYVariable(df): 
    columns = df.columns.tolist()
    
    yVariable = st.selectbox(
        "What variable do you want to plot as your dependent variable",
        (columns)
    )
    st.write(f"You chose: {yVariable} as your dependent variable")
    return yVariable

def plotData(df, x, y, option):
    st.write('Graph below')

    fig, ax = plt.subplots(figsize=(10, 6))  # Create a larger figure

    # Limit the number of data points to plot
    max_points = 1000
    if len(df) > max_points:
        df_sample = df.sample(n=max_points)
    else:
        df_sample = df

    if option == 'Bar Graph':
        ax.bar(df_sample[x], df_sample[y])  # Create bar plot
    elif option == 'Scatter Plot':
        ax.scatter(df_sample[x], df_sample[y], alpha=0.5)  # Create scatter plot with transparency
    elif option == 'Line Plot':
        ax.plot(df_sample[x], df_sample[y])  # Create line plot

    ax.set_xlabel(x)
    ax.set_ylabel(y)
    ax.set_title(f"{option} of {y} vs {x}")

    # Rotate x-axis labels if they're too long
    plt.xticks(rotation=45, ha='right')

    # Adjust layout to prevent cutting off labels
    plt.tight_layout()

    st.pyplot(fig)  # Display the plot in Streamlit

    # Display a message if data was sampled
    if len(df) > max_points:
        st.info(f"Note: The plot shows a random sample of {max_points} points from your data to improve performance.")



# Helper functions for modeling
def train_linear_regression(X_train, X_test, y_train, y_test):
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    return model, mse, r2, y_pred

def train_logistic_regression(X_train, X_test, y_train, y_test):
    model = LogisticRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    return model, accuracy, precision, recall, f1, y_pred

def train_random_forest(X_train, X_test, y_train, y_test, n_estimators, is_classifier):
    if is_classifier:
        model = RandomForestClassifier(n_estimators=n_estimators)
    else:
        model = RandomForestRegressor(n_estimators=n_estimators)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    if is_classifier:
        accuracy = accuracy_score(y_test, y_pred)
        return model, accuracy, y_pred
    else:
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        return model, mse, r2, y_pred

def perform_kmeans(X, n_clusters):
    kmeans = KMeans(n_clusters=n_clusters)
    kmeans.fit(X)
    return kmeans


# Add this function at the beginning of your script, after imports
def set_plot_style():
    # Use dark background style
    plt.style.use('dark_background')

    # Set figure background colors to match the Solarized Dark theme
    mpl.rcParams['figure.facecolor'] = '#002b36'  # base03
    mpl.rcParams['axes.facecolor'] = '#073642'    # base02
    
    # Set text and label colors to match Solarized Dark theme
    mpl.rcParams['text.color'] = '#839496'        # base0
    mpl.rcParams['axes.labelcolor'] = '#93a1a1'   # base1
    mpl.rcParams['xtick.color'] = '#839496'       # base0
    mpl.rcParams['ytick.color'] = '#839496'       # base0
    
    # Set grid color and make it lighter with transparency
    mpl.rcParams['grid.color'] = '#586e75'        # base01
    mpl.rcParams['grid.alpha'] = 0.5              # Lighter gridlines

    # Set figure size and enable grids
    mpl.rcParams['figure.figsize'] = (10, 6)
    mpl.rcParams['axes.grid'] = True
    
    # Increase font sizes for readability
    mpl.rcParams['font.size'] = 12                # General font size
    mpl.rcParams['axes.titlesize'] = 16           # Title font size
    mpl.rcParams['axes.labelsize'] = 14           # Axis labels font size
    mpl.rcParams['xtick.labelsize'] = 12          # X-tick labels font size
    mpl.rcParams['ytick.labelsize'] = 12          # Y-tick labels font size


# Call this function at the beginning of your script
set_plot_style()

    
# Add a main page introducing the app
def main_page():
    st.title("Welcome to InsightFlow!")
    st.write(
        """
        InsightFlow is an interactive tool to help you upload, clean, analyze, and model datasets efficiently. 
        Navigate through the different steps using the sidebar to explore your data and gain insights.
        """
    )


# Multi-page app structure using tabs for navigation
st.sidebar.title("InsightFlow")
pages = st.sidebar.radio("Navigation", ['Main Page', 'Upload & Inspect', 'Clean Data', 'Analyze Data', 'Model Data'])

# Initialize session state for dataset
if 'dataset' not in st.session_state:
    st.session_state['dataset'] = None

if 'cleaned_dataset' not in st.session_state:
    st.session_state['cleaned_dataset'] = None

# Initialize if not already set
if 'clean_option' not in st.session_state:
    st.session_state['clean_option'] = None

# Main Page
if pages == 'Main Page':
    main_page()


# Page 1: Upload and Inspect
if pages == 'Upload & Inspect':
    st.title("Upload and Inspect Data")

    dataset = st.file_uploader('Upload your dataset here', type=['csv'])


    # Only load the dataset if it is uploaded
    if dataset is not None and dataset.name.endswith('.csv'):
        # Load and store dataset in session state
        st.session_state['dataset'] = load_dataset(dataset)

    # Check if dataset is already stored in session_state
    if st.session_state['dataset'] is not None:
        dataset_df = st.session_state['dataset']
        # Display basic dataset information
        display_basic_info(dataset_df)

        if st.button("Analyze Dataset", key="analyze_button"):
            display_dataset_analysis(dataset_df)
    else:
        st.warning("Please upload a valid CSV file.")

# Page 2: Clean Data
elif pages == 'Clean Data':
    st.title("Clean Dataset")
    # Create two columns: one for inputs (left) and one for the graph (right)
    col1, col2 = st.columns([1, 1])  # Adjust column width ratio (1:3)
    
    with col1:
        if st.button('Reset Dataset', key="reset_button"):
            st.session_state['cleaned_dataset'] = None
            st.session_state['clean_option'] = None
            st.rerun()
 
        if st.session_state['dataset'] is not None:
            if st.session_state['cleaned_dataset'] is not None:
                dataset_df = st.session_state['cleaned_dataset']
            else:
                dataset_df = st.session_state['dataset'].copy()
            
            tab1, tab2, tab3 = st.tabs(['Handle Missing Values', 'Handle Duplicated Values', 'Handle Improper Formatting'])
            
            with tab1:
                dataset_df = clean_dataset_null(dataset_df)
            
            with tab2:
                dataset_df = clean_dataset_duplicated(dataset_df)
            
            with tab3:
                dataset_df = clean_dataset_formatting(dataset_df)
            
            with col2:
                st.session_state['cleaned_dataset'] = dataset_df
                st.subheader('Updated Dataset')
                st.dataframe(dataset_df.head(), use_container_width=True)
                display_errors(dataset_df)
        
        else:
            st.error("Please upload and inspect the dataset first on the 'Upload & Inspect' page.")

# Page 3: Analyze Data
elif pages == 'Analyze Data':
    st.title("Analyze Data")

    # Create two columns: one for inputs (left) and one for the graph (right)
    col1, col2 = st.columns([1, 3])  # Adjust column width ratio (1:3)

    # Access dataset from session_state
    if st.session_state['dataset'] is not None:
        dataset_df = st.session_state['cleaned_dataset'] if st.session_state['cleaned_dataset'] is not None else st.session_state['dataset']
        
        # Left column: User input (controls)
        with col1:
            option = chooseOption()  # Select the type of graph (Bar, Scatter, etc.)
            x_variable = chooseXVariable(dataset_df)  # Select the X variable
            y_variable = chooseYVariable(dataset_df)  # Select the Y variable

        # Right column: Graph display
        with col2:
            plotData(dataset_df, x_variable, y_variable, option)  # Generate and display the graph

    else:
        st.error("Please upload and clean the dataset first.")

        st.error("Please upload and clean the dataset first.")

# Page 4: Model Data
elif pages == 'Model Data':
    st.title("Model Data")

    # Check dataset availability
    if 'dataset' in st.session_state and st.session_state['dataset'] is not None:
        dataset_df = st.session_state['cleaned_dataset'] if st.session_state['cleaned_dataset'] is not None else st.session_state['dataset']
        
        # Create two columns
        col1, col2 = st.columns([1, 2])
        
        with col1:
            # Model selection
            model_type = st.selectbox(
                "Select a machine learning model",
                ("Linear Regression", "Logistic Regression", "Random Forest", "K-Means Clustering")
            )

            # Variable selection
            X_columns = st.multiselect("Select independent variables (X)", dataset_df.columns)
            
            if model_type != "K-Means Clustering":
                y_column = st.selectbox("Select dependent variable (Y)", dataset_df.columns)

            # Hyperparameter settings
            if model_type == "Random Forest":
                n_estimators = st.slider("Number of estimators (trees)", 10, 500, 100)
            elif model_type == "K-Means Clustering":
                n_clusters = st.slider("Number of clusters (k)", 2, 10, 3)

            # Train/Test split
            test_size = st.slider("Test set size (%)", 10, 50, 20) / 100

            # Cross-validation option
            do_cv = st.checkbox("Perform cross-validation")

            train_model = st.button("Train Model")

        with col2:
            if train_model:
                if len(X_columns) == 0:
                    st.error("Please select at least one independent variable.")
                elif model_type != "K-Means Clustering" and y_column is None:
                    st.error("Please select a dependent variable.")
                else:
                    X = dataset_df[X_columns]
                    
                    if model_type != "K-Means Clustering":
                        y = dataset_df[y_column]
                        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

                    # Standardize the features
                    scaler = StandardScaler()
                    X_scaled = scaler.fit_transform(X)
                    
                    if model_type != "K-Means Clustering":
                        X_train_scaled = scaler.transform(X_train)
                        X_test_scaled = scaler.transform(X_test)

                    if model_type == "Linear Regression":
                        model, mse, r2, y_pred = train_linear_regression(X_train_scaled, X_test_scaled, y_train, y_test)
                        st.write(f"Mean Squared Error: {mse:.4f}")
                        st.write(f"R² Score: {r2:.4f}")

                        # Plot actual vs predicted values
                        fig, ax = plt.subplots()
                        ax.scatter(y_test, y_pred, alpha=0.5, color='#268bd2')  # blue
                        ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2, color='#dc322f')  # red
                        ax.set_xlabel("Actual")
                        ax.set_ylabel("Predicted")
                        ax.set_title("Actual vs Predicted Values")
                        st.pyplot(fig)

                    elif model_type == "Logistic Regression":
                        model, accuracy, precision, recall, f1, y_pred = train_logistic_regression(X_train_scaled, X_test_scaled, y_train, y_test)
                        st.write(f"Accuracy: {accuracy:.4f}")
                        st.write(f"Precision: {precision:.4f}")
                        st.write(f"Recall: {recall:.4f}")
                        st.write(f"F1 Score: {f1:.4f}")

                        # Plot confusion matrix
                        cm = confusion_matrix(y_test, y_pred)
                        fig, ax = plt.subplots()
                        sns.heatmap(cm, annot=True, fmt='d', ax=ax, cmap='YlGnBu', cbar=False)
                        ax.set_xlabel("Predicted")
                        ax.set_ylabel("Actual")
                        ax.set_title("Confusion Matrix")
                        st.pyplot(fig)

                    elif model_type == "Random Forest":
                        is_classifier = y.dtype == 'object' or y.dtype == 'bool' or len(np.unique(y)) < 10
                        model, performance, y_pred = train_random_forest(X_train_scaled, X_test_scaled, y_train, y_test, n_estimators, is_classifier)
                        
                        if is_classifier:
                            st.write(f"Accuracy: {performance:.4f}")
                            # Plot confusion matrix
                            cm = confusion_matrix(y_test, y_pred)
                            fig, ax = plt.subplots()
                            sns.heatmap(cm, annot=True, fmt='d', ax=ax, cmap='YlGnBu')
                            ax.set_xlabel("Predicted")
                            ax.set_ylabel("Actual")
                            ax.set_title("Confusion Matrix")
                            st.pyplot(fig)
                        else:
                            mse, r2 = performance
                            st.write(f"Mean Squared Error: {mse:.4f}")
                            st.write(f"R² Score: {r2:.4f}")
                            # Plot actual vs predicted values
                            fig, ax = plt.subplots()
                            ax.scatter(y_test, y_pred, alpha=0.5)
                            ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
                            ax.set_xlabel("Actual")
                            ax.set_ylabel("Predicted")
                            ax.set_title("Actual vs Predicted Values")
                            st.pyplot(fig)

                        # Feature importance
                        feature_importance = pd.DataFrame({'feature': X_columns, 'importance': model.feature_importances_})
                        feature_importance = feature_importance.sort_values('importance', ascending=False)
                        st.write("Feature Importance:")
                        fig, ax = plt.subplots()
                        sns.barplot(x='importance', y='feature', data=feature_importance, ax=ax, palette='YlGnBu')
                        ax.set_title("Feature Importance")
                        st.pyplot(fig)

                    elif model_type == "K-Means Clustering":
                        kmeans = perform_kmeans(X_scaled, n_clusters)
                        st.write(f"Inertia: {kmeans.inertia_:.4f}")
                        
                        # Display cluster centers
                        cluster_centers = pd.DataFrame(scaler.inverse_transform(kmeans.cluster_centers_), columns=X_columns)
                        st.write("Cluster Centers:")
                        st.write(cluster_centers)

                        # Display number of points in each cluster
                        cluster_counts = pd.Series(kmeans.labels_).value_counts().sort_index()
                        st.write("Number of points in each cluster:")
                        st.write(cluster_counts)

                        # Plot clustered data (first two features)
                        fig, ax = plt.subplots()
                        scatter = ax.scatter(X_scaled[:, 0], X_scaled[:, 1], c=kmeans.labels_, cmap='viridis', alpha=0.6)
                        ax.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], color='#dc322f', marker='x', s=200, linewidths=3)
                        ax.set_xlabel(X_columns[0])
                        ax.set_ylabel(X_columns[1])
                        ax.set_title("K-Means Clustering Result")
                        plt.colorbar(scatter)
                        st.pyplot(fig)

                    # Cross-validation
                    if do_cv:
                        if model_type != "K-Means Clustering":
                            if model_type == "Linear Regression":
                                cv_model = LinearRegression()
                                cv_metric = "r2"
                            elif model_type == "Logistic Regression":
                                cv_model = LogisticRegression()
                                cv_metric = "accuracy"
                            else:  # Random Forest
                                cv_model = RandomForestRegressor(n_estimators=n_estimators) if not is_classifier else RandomForestClassifier(n_estimators=n_estimators)
                                cv_metric = "r2" if not is_classifier else "accuracy"
                            
                            cv_scores = cross_val_score(cv_model, X_scaled, y, cv=5, scoring=cv_metric)
                            st.write(f"Cross-validation {cv_metric} scores: {cv_scores}")
                            st.write(f"Mean cross-validation {cv_metric} score: {cv_scores.mean():.4f}")
                        else:
                            st.write("Cross-validation is not applicable for K-Means Clustering.")

    else:
        st.error("Please upload and clean the dataset first.")

# Set Matplotlib parameters
mpl.rcParams['agg.path.chunksize'] = 10000

