import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.tree import DecisionTreeRegressor


def load_data(file_path, column_names):
    return pd.read_csv(file_path, names=column_names)


def preprocess_data(data):
    x = data.iloc[:, 2:]
    y = data.iloc[:, -1]

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=1, shuffle=True)

    scaler = StandardScaler()
    scaler.fit(x_train, y_train)

    x_train = scaler.transform(x_train)
    x_test = scaler.transform(x_test)

    y_train = y_train.values.reshape(-1, 1)
    y_test = y_test.values.reshape(-1, 1)

    y_scale = StandardScaler()
    y_scale.fit(y_train)

    y_train = y_scale.transform(y_train)
    y_test = y_scale.transform(y_test)

    return x_train, x_test, y_train, y_test


def train_decision_tree(x_train, y_train):
    regressor = DecisionTreeRegressor()
    regressor.fit(x_train, y_train)
    return regressor


def evaluate_model(model, x_test, y_test):
    y_predict = model.predict(x_test)
    score = model.score(x_test, y_test)
    mse = mean_squared_error(y_test, y_predict)
    r2 = r2_score(y_test, y_predict)

    return score, mse, r2, y_predict


def plot_ground_truth_vs_predicted(y_test, y_predict):
    fig, ax = plt.subplots()

    ax.scatter(y_test, y_predict, edgecolors=(0, 0, 0))
    ax.plot([y_test.min(), y_test.max()], [
            y_test.min(), y_test.max()], 'k--', lw=4)
    ax.set_xlabel('Actual')
    ax.set_ylabel('Predicted')
    ax.set_title("Ground Truth vs Predicted")

    plt.show()


def plot_test_vs_prediction(y_test, y_predict):
    df = pd.DataFrame({'Tests': y_test.flatten(),
                      'Prediction': y_predict.flatten()})
    df.sort_index(inplace=True)
    df.plot(kind='line', figsize=(18, 8))
    plt.grid(which='major', linestyle='-', linewidth='0.5', color='green')
    plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
    plt.show()


if __name__ == "__main__":
    names = ['VENDOR', 'MODEL_NAME', 'MYCT', 'MMIN',
             'MMAX', 'CACH', 'CHMIN', 'CHMAX', 'PRP', 'ERP']
    data = load_data('machine.data', names)
    x_train, x_test, y_train, y_test = preprocess_data(data)

    decision_tree_model = train_decision_tree(x_train, y_train)
    score, mse, r2, y_predict = evaluate_model(
        decision_tree_model, x_test, y_test)

    print("---> Decision Tree Regression <---")
    print("Coefficient of determination R^2 of the prediction.: ", score)
    print("Mean squared error: %.2f" % mse)
    print('Test Variance score: %.2f' % r2)

    plot_ground_truth_vs_predicted(y_test, y_predict)
    plot_test_vs_prediction(y_test, y_predict)
