import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def load_data(file_path: str) -> pd.DataFrame:
    df = pd.read_csv(file_path, sep=';')
    df.columns = [col.strip().lower() for col in df.columns]
    return df

def eda_summary(df: pd.DataFrame) -> None:
    print("Data Shape:", df.shape)
    print("\nMissing Values:\n", df.isnull().sum())
    print("\nTarget Variable Distribution:\n", df['y'].value_counts(normalize=True))
    print("\nCategorical Features:\n", df.select_dtypes(include='object').columns.tolist())
    print("\nNumerical Features:\n", df.select_dtypes(include='number').columns.tolist())

def plot_categorical_features(df: pd.DataFrame):
    cat_vars = df.select_dtypes(include='object').columns
    for var in cat_vars:
        if var != 'y':
            plt.figure(figsize=(6, 4))
            sns.countplot(data=df, x=var, hue='y')
            plt.title(f"Subscription Rate by {var.capitalize()}")
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.show()

def plot_numerical_features(df: pd.DataFrame):
    num_vars = df.select_dtypes(include='number').columns
    for var in num_vars:
        plt.figure(figsize=(6, 4))
        sns.histplot(data=df, x=var, hue='y', kde=True, bins=30)
        plt.title(f"{var.capitalize()} Distribution by Subscription")
        plt.tight_layout()
        plt.show()

def correlation_with_target(df: pd.DataFrame):
    df_encoded = df.copy()
    df_encoded['y'] = df_encoded['y'].map({'yes': 1, 'no': 0})
    corr = df_encoded.corr(numeric_only=True)['y'].sort_values(ascending=False)
    print("\nCorrelation with target variable:\n", corr)

def run_eda(file_path: str):
    df = load_data(file_path)
    eda_summary(df)
    plot_categorical_features(df)
    plot_numerical_features(df)
    correlation_with_target(df)

if __name__ == "__main__":
    import sys
    file_path = sys.argv[1]
    run_eda(file_path)
