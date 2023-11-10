# You are not allowed to import any additional packages/libraries.
import numpy as np
from pandas import DataFrame, read_csv
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

class LogisticRegression:
    def __init__(self, learning_rate=0.0001, iteration=5000000):
        self.learning_rate = learning_rate
        self.iteration = iteration
        self.weights = None
        self.intercept = None

    # This function computes the gradient descent solution of logistic regression.
    def fit(self, X, y):
        #Initialize weights and intercept
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.intercept = 0


        for _ in range(self.iteration):
            linear_model = np.dot(X, self.weights) + self.intercept
            y_pred = self.sigmoid(linear_model)
            # if _ % 50000 == 0:
            #     print(np.mean(y_pred - y))

            dw = (1 / n_samples) * np.dot(X.T, (y_pred - y))
            db = (1 / n_samples) * np.sum(y_pred - y)

            self.weights -= self.learning_rate * dw
            self.intercept -= self.learning_rate * db
                    
    # This function takes the input data X and predicts the class label y according to your solution.
    def predict(self, X):
        linear_model = np.dot(X, self.weights) + self.intercept
        y_pred = self.sigmoid(linear_model)
        return [1 if i > 0.5 else 0 for i in y_pred]

    # This function computes the value of the sigmoid function.
    def sigmoid(self, x):
        return 1 /(1 + np.exp(-x))
        

class FLD:
    def __init__(self):
        self.w = None
        self.m0 = None
        self.m1 = None
        self.sw = None
        self.sb = None
        self.slope = None

    # This function computes the solution of Fisher's Linear Discriminant.
    def fit(self, X, y):
        X0 = X[y==0]
        X1 = X[y==1]
        self.m0 = np.mean(X0, axis=0)
        self.m1 = np.mean(X1, axis=0)
        self.sw = np.dot((X0 - self.m0).T, (X0 - self.m0)) + np.dot((X1 - self.m1).T, (X1 - self.m1))
        
        self.sb = np.outer(self.m1 - self.m0, self.m1 - self.m0)
        eigvals, eigvecs = np.linalg.eig(np.linalg.inv(self.sw).dot(self.sb))
        self.w = eigvecs[:, np.argmax(eigvals)]


    # This function takes the input data X and predicts the class label y by comparing the distance between the projected result of the testing data with the projected means (of the two classes) of the training data.
    # If it is closer to the projected mean of class 0, predict it as class 0, otherwise, predict it as class 1.
    def predict(self, X):
        projected_data = np.dot(X, self.w)
        projected_mean_0 = np.dot(self.m0, self.w)
        projected_mean_1 = np.dot(self.m1, self.w)

        y_pred = []
        for data_point in projected_data:
            distance_to_mean_0 = abs(data_point - projected_mean_0)
            distance_to_mean_1 = abs(data_point - projected_mean_1)

            if distance_to_mean_0 < distance_to_mean_1:
                y_pred.append(0)
            else:
                y_pred.append(1)

        return np.array(y_pred)
        

    # This function plots the projection line of the testing data.
    # You don't need to call this function in your submission, but you have to provide the screenshot of the plot in the report.
    def plot_projection(self, X, file_name):
        
        fig, ax = plt.subplots(figsize=(10,10))
        colors=['red','blue']
        y_pred = FLD.predict(X)
        slope = self.w[1]/self.w[0] # compute slope
        x = np.linspace(-60,-10,700)
        y = slope*x+0
        plt.plot(x,y, '-g')
        
        for point, pred in zip(X, y_pred):
            point_project = (np.dot(point,self.w)*self.w) / np.dot(self.w.T, self.w) # project point
            # draw points and lines
            plt.title(f"Projection Line: slope={slope}, intercept=0")
            plt.plot( [point_project[0], point[0]],[point_project[1], point[1]], color='black', linewidth=0.1,zorder=1) 
            plt.scatter(point_project[0], point_project[1], color=colors[pred], s=5, zorder=2)
            plt.scatter(point[0],point[1],color=colors[pred], s=5, zorder=3)
        plt.savefig(f'{file_name}.png')

     
# Do not modify the main function architecture.
# You can only modify the value of the arguments of your Logistic Regression class.
if __name__ == "__main__":
# Data Loading
    train_df = DataFrame(read_csv("train.csv"))
    test_df = DataFrame(read_csv("test.csv"))

# Part 1: Logistic Regression
    # Data Preparation
    # Using all the features for Logistic Regression
    X_train = train_df.drop(["target"], axis=1)
    y_train = train_df["target"]
    X_test = test_df.drop(["target"], axis=1)
    y_test = test_df["target"]
    
    X_train = X_train.to_numpy()
    y_train = y_train.to_numpy()
    X_test = X_test.to_numpy()
    y_test = y_test.to_numpy()

    # Model Training and Testing
    LR = LogisticRegression(learning_rate=0.00001, iteration=3000000)
    LR.fit(X_train, y_train)
    y_pred = LR.predict(X_test)
    accuracy = accuracy_score(y_test , y_pred)
    print(f"Part 1: Logistic Regression")
    print(f"Weights: {LR.weights}, Intercept: {LR.intercept}")
    print(f"Accuracy: {accuracy}")
    # You must pass this assertion in order to get full score for this part.
    assert accuracy > 0.75, "Accuracy of Logistic Regression should be greater than 0.75"

# Part 2: Fisher's Linear Discriminant
    # Data Preparation
    # Only using two features for FLD
    X_train = train_df[["age", "thalach"]]
    y_train = train_df["target"]
    X_test = test_df[["age", "thalach"]]
    y_test = test_df["target"]
    
    X_train = X_train.to_numpy()
    y_train = y_train.to_numpy()
    X_test = X_test.to_numpy()
    y_test = y_test.to_numpy()

    # Model Training and Testing
    FLD = FLD()
    FLD.fit(X_train, y_train)
    y_pred = FLD.predict(X_test)
    accuracy = accuracy_score(y_test , y_pred)
    print(f"Part 2: Fisher's Linear Discriminant")
    print(f"Class Mean 0: {FLD.m0}, Class Mean 1: {FLD.m1}")
    print(f"With-in class scatter matrix:\n{FLD.sw}")
    print(f"Between class scatter matrix:\n{FLD.sb}")
    print(f"w:\n{FLD.w}")
    print(f"Accuracy of FLD: {accuracy}")
    # You must pass this assertion in order to get full score for this part.
    assert accuracy > 0.65, "Accuracy of FLD should be greater than 0.65"
    FLD.plot_projection(X_train, 'train')
    FLD.plot_projection(X_test, 'test')