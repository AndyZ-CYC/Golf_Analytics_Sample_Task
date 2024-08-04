import pandas as pd
import numpy as np
import statsmodels.api as sm

def read_data(file_path, target_column_ids, target_column_names, file_delimiter = ';'):
    """
    Reads data from a text file and returns the columns of interest with the updated column names

    Parameters:
    - file_path: str, path to the text file
    - target_column_ids: list of int, indices of the target columns
    - target_column_names: list of str, names of the target columns
    - file_delimiter: str, delimiter of the txt file, default as ';'

    Returns:
    - DataFrame containing the selected columns
    """
    # read the txt file with read_csv
    data = pd.read_csv(file_path, delimiter=file_delimiter)
    # select target columns
    selected_columns = data.iloc[:, target_column_ids]
    selected_columns.columns = target_column_names

    return selected_columns


def OLS_regression(data, X_columns, y_column):
    """
    Performs linear regression on the given data

    Parameters:
    - data: DataFrame containing input data
    - X_columns: list of str, containing column names of independent variables
    - y_columns: str, column name of target variable

    Returns:
    - Regression results (summary)
    """
    # form X and y matrices
    X = data[X_columns]
    X = sm.add_constant(X)    # Adds a contant term
    y = data[y_column]
    # run the regression
    model = sm.OLS(y, X).fit()

    return model.summary()


def monte_carlo_simulation(data, target_column, n_trials=10000, sample_count=4, threshold=270, random_seed=42):
    """
    Perform Monte Carlo Simulation to estimate the probability that the sum of given number of values in the 
    target column is less than or equal to the threshold

    Parameters:
    - data: DataFrame that contains the target column
    - target_column: str, name of the target column
    - n_trials: int,  number of simulation trails
    - sample_count: int, number of sample in each simulation trials
    - threshold: int, summation threshold
    - random_seed: int, random seed for reproducibility
    
    Returns:
    - Two float numbers containing the estimated probability and standard error
    """
    # set random seed
    np.random.seed(random_seed)

    # get sample values
    scores = data[target_column].values

    # Generate simulation array 
    random_samples = np.random.choice(scores, size=(n_trials, sample_count), replace=True)

    # Calculate summation
    sum_of_samples = random_samples.sum(axis=1)

    # Calculate estimated probability of sums less than or equal to the threshold
    prob_estimated = np.mean(sum_of_samples <= threshold)

    # Calculate standard error
    standard_error = np.sqrt(prob_estimated * (1 - prob_estimated) / n_trials)

    return prob_estimated, standard_error


def output_results(file_path, regression_result, probability, standard_error):
    """
    Writes the regression result and the simulation output to a text file

    Parameters:
    - file_path: str, path to the output text file
    - regression_results: regression results summary
    - probability: float, estimated probability of the Monte Carlo Simulation
    - standard_error: float, standard error of the Monte Carlo Simulation
    """
    with open(file_path, 'w') as f:
        f.write("Regression Analysis Results:\n")
        f.write(regression_result.as_text())
        f.write("\n\n")
        f.write("Monte Carlo Simulation Results:\n")
        f.write(f"Estimated Probability: {probability:.4f}\n")
        f.write(f"Standard Error: {standard_error:.4f}\n")


if __name__ == "__main__":
    # Parameters for read data
    file_path = './data/round-2014-small.txt'
    file_delimiter = ';'
    target_column_ids = [15, 79, 122]
    target_column_names = ['Score', 'GIR', 'Putt']
    # Parameters for regression
    OLS_X_columns = ['GIR', 'Putt']
    OLS_y_column = 'Score'
    # Parameters for simulation
    simulation_target_column = 'Score'
    n_trials=10000
    sample_count=4
    threshold=270
    random_seed=42
    # Parameters for output results
    output_path = './output.txt'

    data = read_data(file_path, target_column_ids, target_column_names, file_delimiter)
    regression_result = OLS_regression(data, OLS_X_columns, OLS_y_column)
    probability, standard_error = monte_carlo_simulation(data, simulation_target_column, n_trials, sample_count, threshold, random_seed)
    output_results(output_path, regression_result, probability, standard_error)


