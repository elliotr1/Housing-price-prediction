import pandas as pd
from scipy import stats
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn import linear_model
import numpy as np
from matplotlib import pyplot
def linear_regression_model(x, y):
    """
    Trains a linear regression model and evaluates its performance using metrics.

    Parameters:
    - x: The feature matrix.
    - y: The target vector.

    Returns:
    A list containing Mean Squared Error and Mean Absolute Error.
    """
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x)
    X_train, X_test, y_train, y_test = train_test_split(x_scaled, y, test_size=0.25, shuffle=True)
    reg = linear_model.LinearRegression()
    reg.fit(X_train, y_train)
    pred = reg.predict(X_test)
    MSE = mean_squared_error(y_test, pred)
    MAE = mean_absolute_error(y_test, pred)
    r2 = r2_score(y_test, pred)
    pyplot.xlabel = "Actual"
    pyplot.ylabel = "Predicted"
    pyplot.scatter(y_test, pred)
    pyplot.show()

    print("MSE: ", MSE)
    print("MAE: ", MAE)
    print("R^2: ", r2)
    return [MSE, MAE]

def statistical_analysis(df):
    """
    Provides a statistical overview of the dataset.

    Parameters:
    - df: The DataFrame to analyze.
    """
    analysis_df = pd.DataFrame()
    analysis_df["column names"] = df.columns
    analysis_df["min"] = [np.min(df[column]) for column in df.columns]
    analysis_df["max"] = [np.max(df[column]) for column in df.columns]
    analysis_df["mean"] = [np.mean(df[column]) for column in df.columns]
    analysis_df["std"] = [np.std(df[column]) for column in df.columns]
    analysis_df["prec_25"] = [np.percentile(df[column], 25) for column in df.columns]
    analysis_df["prec_50"] = [np.percentile(df[column], 50) for column in df.columns]
    analysis_df["prec_75"] = [np.percentile(df[column], 75) for column in df.columns]
    print(analysis_df.to_string())

def normalize_and_clean_data(df):
    """
    Performs data normalization and cleaning.

    Parameters:
    - df: The DataFrame to process.

    Returns:
    The processed DataFrame.
    """
    df = df.drop_duplicates()
    encoder = LabelEncoder()
    object_columns = df.select_dtypes(include=["object"]).columns
    df[object_columns] = df[object_columns].apply(encoder.fit_transform)
    df['price'] = df['price'] / 20000000
    df["area"] = df["area"] / 1000
    z_scores = np.abs(stats.zscore(df))
    threshold = 2
    df = df[(z_scores <= threshold).all(axis=1)]
    return df

def pearson_coefficient(x):
    """
    Calculates Pearson's correlation coefficient between different columns.

    Parameters:
    - x: The feature matrix.
    """
    correlation_matrix = x.corr()
    print("Pearson's Correlation between different columns: ")
    print(correlation_matrix.to_string())
def most_valued_columns(x, y):
    """
    Identifies the most valuable columns by evaluating model performance with one column removed.

    Parameters:
    - x: The feature matrix.
    - y: The target vector.
    """
    mse = []
    for column in x.columns:
        print(f"Current column being removed: {column}")
        mse_r, _ = linear_regression_model(x.drop(columns=[column]), y)
        mse.append(mse_r)
    print("Most valuable column:", list(x.columns)[mse.index(max(mse))])

def main():
    df = pd.read_csv("Datasets/Housing.csv")
    df = normalize_and_clean_data(df)

    # Perform your analysis steps
    statistical_analysis(df)
    pearson_coefficient(df.drop(columns=["price", "guestroom", "hotwaterheating"]))
    most_valued_columns(df.drop(columns=["price", "guestroom", "hotwaterheating"]), df["price"])
    linear_regression_model(df.drop(columns=["price", "guestroom", "hotwaterheating"]), df["price"])

if __name__ == "__main__":
    main()
