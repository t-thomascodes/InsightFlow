# InsightFlow

Interactive Streamlit app for dataset analysis and machine learning. Upload CSVs, clean data, visualize patterns, and train models without writing code.

## What This Does

- **Upload & Inspect**: Load CSV datasets and view basic statistics, column info, and data quality metrics
- **Clean Data**: Handle missing values, remove duplicates, fix formatting issues with multiple strategies
- **Analyze Data**: Generate bar graphs, scatter plots, line plots, and pie charts with variable selection
- **Model Data**: Train ML models with cross-validation, hyperparameter tuning, and performance metrics

## Features

### Data Cleaning
- Multiple strategies for handling null values (mean/median fill, drop rows/columns)
- Duplicate detection and removal
- Automatic string formatting cleanup
- Real-time dataset preview after each operation

### Visualization
- Interactive graph generation with variable selection
- Support for large datasets (automatic sampling for performance)
- Customizable plot types: bar, scatter, line, pie charts
- Solarized Dark theme for reduced eye strain

### Machine Learning Models
- **Linear Regression**: MSE, R² scores with actual vs predicted plots
- **Logistic Regression**: Accuracy, precision, recall, F1, confusion matrix
- **Random Forest**: Classification/regression with feature importance analysis
- **K-Means Clustering**: Inertia metrics, cluster visualization, center analysis

### Model Features
- Train/test split configuration
- Feature standardization
- Cross-validation support
- Interactive hyperparameter tuning
- Visual performance metrics

## Setup
```bash
# Clone the repo
git clone https://github.com/yourusername/insightflow.git
cd insightflow

# Install dependencies
pip install streamlit pandas matplotlib seaborn scikit-learn numpy

# Run the app
streamlit run insightflow.py
```

Open `http://localhost:8501` in your browser.

## Usage

1. **Upload Dataset**: Navigate to "Upload & Inspect" and upload a CSV file
2. **Clean Data**: Use "Clean Data" tab to handle missing values, duplicates, formatting
3. **Visualize**: Select variables and plot types in "Analyze Data"
4. **Model**: Choose ML algorithm, select features, configure hyperparameters in "Model Data"

## Technical Details

- **Framework**: Streamlit for interactive UI
- **Visualization**: Matplotlib, Seaborn with Solarized Dark theme
- **ML**: scikit-learn for models and preprocessing
- **Data Processing**: Pandas for dataset manipulation
