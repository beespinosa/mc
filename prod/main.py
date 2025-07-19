import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import xgboost as xgb
import shap
import matplotlib.pyplot as plt

def load_data(file_path: str):
    """
    Loads a CSV dataset using semicolon as the separator.

    Parameters:
    - file_path: Path to the CSV file

    Returns:
    - df: Loaded DataFrame with the contents of the file
    """
    df = pd.read_csv(file_path, sep=';')
    return df

def xgb_preprocess_data(df: pd.DataFrame):
    """
    Preprocesses a DataFrame for XGBoost training by handling target encoding, feature cleanup,
    and categorical type conversion.

    Parameters:
    - df: Input DataFrame containing features and the target column 'y'

    Returns:
    - df: Preprocessed feature DataFrame
        - Filters out rows where 'duration' == 0
        - Drops the 'duration' column to prevent data leakage
        - Converts object-type columns to 'category' dtype for XGBoost compatibility
    - y: Binary-encoded target series (1 for 'yes', 0 for 'no')
    - categorical_cols: List of categorical column names
    """
    df = df.copy()
    # Remove cero time calls for noise reduction
    df = df[df['duration'] > 0]  
    y = df.pop('y').map({'yes': 1, 'no': 0})

    # Drop duration feature to avoid leakage
    if 'duration' in df.columns:
        df.drop(columns='duration', inplace=True)

    categorical_cols = df.select_dtypes(include='object').columns.tolist()
    for colname in categorical_cols:
        df[colname] = df[colname].astype('category')

    return df, y, categorical_cols

def xgb_train_model(X, y):
    """
    Trains an XGBoost classification model and evaluates its performance on a test set.

    Parameters:
    - X: Feature matrix (numeric and categorical)
    - y: Target vector (binary classification)

    Returns:
    - model: Trained XGBoost classifier
    - X_test: Test feature matrix (for further evaluation and explanation)
    - y_test: True labels for the test set

    Additional Output:
    - Prints a classification report (precision, recall, f1-score) and a confusion matrix for test predictions.
    """
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

    model = xgb.XGBClassifier(tree_method="hist", early_stopping_rounds=2, enable_categorical=True)
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)])
    model.save_model("./outputs/model.json")
    y_pred = model.predict(X_test)

    print("\nModel Performance:\n")
    print(classification_report(y_test, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

    return model, X_test, y_test

def xgb_explain_model(model, X, save_path="outputs/visuals/shap_beeswarm.png"):
    """
    Generates and saves a SHAP beeswarm plot to explain feature importance for the XGBoost model.

    Parameters:
    - model: Trained XGBoost model (xgb.XGBClassifier)
    - X: Feature matrix (same format used during training)
    - save_path: File path to save the SHAP plot

    Returns:
    - feature_importance: List of feature names ordered by their mean absolute SHAP value (descending)
    """
    explainer = shap.TreeExplainer(model)
    explanation = explainer(X)

    # Ensure output directory exists
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # Generate and save SHAP summary plot
    shap.summary_plot(explanation.values, features=X, feature_names=X.columns, show=False)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()

    # Get mean absolute SHAP values and sort features
    shap_values = np.abs(explanation.values).mean(axis=0)
    feature_names = explanation.feature_names
    feature_importance = sorted(zip(feature_names, shap_values), key=lambda x: x[1], reverse=True)

    # Return ordered list of feature names
    return [f for f, _ in feature_importance]

def categorical_summary(df: pd.DataFrame, feature: str, target: str):
    """
    Returns a summary by a specific category with counts and percentages based on a binary 'yes'/'no' target.

    Parameters:
    - df: Input DataFrame
    - feature: name of the categorical column to analyze
    - target: name of the target column with values 'yes' and 'no'

    Returns:
    - DataFrame with the following columns:
        N_total: total number of records per category
        N_yes: number of records with target = 'yes'
        N_no: number of records with target = 'no'
        pct_yes_total: N_yes / N_total (percentage of 'yes' over the total)
        pct_yes_no: N_yes / N_no (ratio of 'yes' to 'no')
    """
    grouped = df.groupby(feature)[target].value_counts().unstack(fill_value=0)

    grouped.columns = grouped.columns.astype(str)
    grouped['N_total'] = grouped.sum(axis=1)
    grouped['N_yes'] = grouped.get('yes', 0)
    grouped['N_no'] = grouped.get('no', 0)

    grouped['pct_yes_total'] = grouped['N_yes'] / grouped['N_total']
    grouped['pct_yes_no'] = grouped.apply(lambda row: row['N_yes'] / row['N_no'] if row['N_no'] > 0 else None, axis=1)

    return grouped[['N_total', 'N_yes', 'N_no', 'pct_yes_total', 'pct_yes_no']].sort_values('pct_yes_total', ascending=False)

if __name__ == "__main__":
    import sys
    file_path = sys.argv[1]
    df = load_data(file_path)
    X, y, categorical_cols = xgb_preprocess_data(df)
    model, X_test, y_test = xgb_train_model(X, y)
    feature_importance = xgb_explain_model(model, X_test)
    
    # Usage of categorical_summary for the first 5 categorical features
    N = 5
    feature_importance_cat = [feature for feature in feature_importance if feature in categorical_cols]
    for index in range(N):
        feature = feature_importance_cat[index]
        if feature in categorical_cols:
            print(f"\nSummary for {feature}:\n")
            summary_table = categorical_summary(df, feature=feature, target='y')
            print(summary_table)