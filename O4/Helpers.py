
from sklearn.model_selection import train_test_split
def split_data(data):
    # Split data into features and target
    X = data.drop('EstimatedSalary', axis=1)
    Y = data['EstimatedSalary']

    # Split data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test
