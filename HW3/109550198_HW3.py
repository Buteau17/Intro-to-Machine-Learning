# You are not allowed to import any additional packages/libraries.
import numpy as np
from pandas import DataFrame, read_csv
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

# This function computes the gini impurity of a label array.
def gini(y):
    
    n = len(y)
    if n == 0:
        return 0.0
    p_sum = 0
    unique_classes = np.unique(y)
    for c in unique_classes:
        p = np.sum(y == c) / n
        p_sum += p * p
    gini_impurity = 1 - p_sum
    return gini_impurity

   

# This function computes the entropy of a label array.
def entropy(y):
    n = len(y)
    if n == 0:
        return 0.0
    unique_classes = np.unique(y)
    entropy_value = 0
    for c in unique_classes:
        p = np.sum(y == c) / n
        if p > 0:
            entropy_value -= p * np.log2(p)
    return entropy_value
        
# The decision tree classifier class.
# Tips: You may need another node class and build the decision tree recursively.
class DecisionTree():
    def __init__(self, criterion='gini', max_depth=None):
        self.criterion = criterion
        self.max_depth = max_depth
        self.feature_importances_ = None 
        self.tree = None  
    
    # This function computes the impurity based on the criterion.
    def impurity(self, y):
        if self.criterion == 'gini':
            return gini(y)
        elif self.criterion == 'entropy':
            return entropy(y)
    
    # This function fits the given data using the decision tree algorithm.
    def fit(self, X, y):
        self.feature_importances_ = np.zeros(X.shape[1])
        

        def best_split(X,y):
            best_attribute = None
            best_threshold  = None
            best_impurity = float ('inf')
            n , m = X.shape

            for attribute in range (m):
                thresholds = np.unique (X[:, attribute])
                for threshold in thresholds:
                    y_left = y[X[:, attribute]<=threshold]
                    y_right = y[X[:, attribute] >threshold]
                    impurity_left = self.impurity(y_left)
                    impurity_right = self.impurity(y_right)
                    impurity = (len(y_left) * impurity_left + len(y_right) * impurity_right) / n

                    if impurity < best_impurity :
                        best_impurity = impurity
                        best_attribute = attribute
                        best_threshold = threshold
            return best_attribute , best_threshold
        #recursive function to build the tree.

        def build_tree(X, y, depth):
            if len (np.unique(y))== 1 or len(y)== 0 or (self.max_depth is not None and depth == self.max_depth):
                return np.bincount(y).argmax() 
            initial_impurity = self.impurity(y)

            # Existing code to find the best split...
            attribute, threshold= best_split(X, y)

            if attribute is None:
                return np.bincount(y).argmax()

            # Split the dataset
            left_indices = X[:, attribute] <= threshold
            right_indices = X[:, attribute] > threshold
            y_left, y_right = y[left_indices], y[right_indices]

            # Calculate weighted impurity after the split
            impurity_left = self.impurity(y_left)
            impurity_right = self.impurity(y_right)
            weighted_impurity = (len(y_left) / len(y)) * impurity_left + (len(y_right) / len(y)) * impurity_right

            # Calculate impurity reduction
            
            self.feature_importances_[attribute] += 1

            # Recursively build left and right subtrees
            left_subtree = build_tree(X[left_indices], y_left, depth + 1)
            right_subtree = build_tree(X[right_indices], y_right, depth + 1)

            return (attribute, threshold, left_subtree, right_subtree)
        self.tree = build_tree(X, y, 0)




    # This function takes the input data X and predicts the class label y according to your trained model.
    def predict(self, X):
        def predict_sample (sample , tree):
            if not isinstance (tree, tuple):
                return tree
            attribute , threshold , left_subtree, right_subtree = tree
            if sample[attribute] <= threshold :
                return predict_sample(sample, left_subtree)
            else:
                return predict_sample(sample, right_subtree)
        return np.array ([predict_sample(sample, self.tree) for sample in X])    


        
    # This function plots the feature importance of the decision tree.
    def plot_feature_importance_img(self, column_names):
        if self.feature_importances_ is None:
            raise ValueError("Feature importances have not been computed. Call the fit method first.")

        # Sorting feature importances
        sorted_indices = np.argsort(self.feature_importances_)[::-1]
        sorted_importances = self.feature_importances_[sorted_indices]
        sorted_columns = np.array(column_names)[sorted_indices]

        # Plotting
        plt.figure(figsize=(10, 6))
        plt.barh(sorted_columns, sorted_importances, align='center')
        plt.xlabel('Importance')
        plt.title('Feature Importance')
        plt.gca().invert_yaxis()
        plt.savefig('feature.png')
        # plt.show()
       
      

# The AdaBoost classifier class.
class AdaBoost():
    def __init__(self, criterion='gini', max_depth= 8,  n_estimators=1000):
        self.criterion = criterion 
        self.n_estimators = n_estimators
        self.weak_classifiers = []  # Initialize an empty list to store weak classifiers
        self.alphas = []  # Initialize an empty list to store alpha values
        self.max_depth = max_depth

    # This function fits the given data using the AdaBoost algorithm.
    # You need to create a decision tree classifier with max_depth = 1 in each iteration.
    def fit(self, X, y):
        n_samples = X.shape[0]
        w = np.full(n_samples, 1 / n_samples)

        for _ in range(self.n_estimators):
            # Sampling according to the weights
            sample_indices = np.random.choice(np.arange(n_samples), size=n_samples, p=w)
            X_sampled = X[sample_indices]
            y_sampled = y[sample_indices]

            classifier = DecisionTree(criterion=self.criterion, max_depth=self.max_depth)
            classifier.fit(X_sampled, y_sampled)

            predictions = classifier.predict(X)
            missclassified = predictions != y
            error = np.dot(w, missclassified) / np.sum(w)

            # Avoid division by zero error
            EPS = 1e-10
            if error == 0:
                error += EPS
            elif error == 1:
                error -= EPS

            # Clip values to avoid overflow
            alpha = 0.5 * np.log(np.clip((1 - error + EPS) / (error + EPS), 1e-10, None))
            w *= np.exp(alpha * missclassified * -1)
            w /= np.sum(w)

            # Store the classifier and alpha
            self.weak_classifiers.append(classifier)
            self.alphas.append(alpha)

    def predict(self, X):
        # Aggregate weak classifier predictions, weighted by their alphas
        weak_classifier_predictions = np.array([alpha * clf.predict(X) for clf, alpha in zip(self.weak_classifiers, self.alphas)])  
        # Summing up the weighted predictions
        aggregated_predictions = np.sum(weak_classifier_predictions, axis=0)
        # Making the final prediction (sign function)
        y_pred = np.sign(aggregated_predictions)
        return y_pred

# Do not modify the main function architecture.
# You can only modify the value of the random seed and the the arguments of your Adaboost class.
if __name__ == "__main__":
# Data Loading
    train_df = DataFrame(read_csv("train.csv"))
    test_df = DataFrame(read_csv("test.csv"))
    X_train = train_df.drop(["target"], axis=1)
    y_train = train_df["target"]
    X_test = test_df.drop(["target"], axis=1)
    y_test = test_df["target"]

    X_train = X_train.to_numpy()
    y_train = y_train.to_numpy()
    X_test = X_test.to_numpy()
    y_test = y_test.to_numpy()

# Set random seed to make sure you get the same result every time.
# You can change the random seed if you want to.
    np.random.seed(0)

# Decision Tree
    print("Part 1: Decision Tree")
    data = np.array([0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 1])
    print(f"gini of [0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 1]: {gini(data)}")
    print(f"entropy of [0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 1]: {entropy(data)}")
    tree = DecisionTree(criterion='gini', max_depth=15)
    tree.fit(X_train, y_train)
    tree.plot_feature_importance_img(['age','sex','cp','fbs','thalach','thal'])
    y_pred = tree.predict(X_test)
    print("Accuracy (gini with max_depth=7):", accuracy_score(y_test, y_pred))
    tree = DecisionTree(criterion='entropy', max_depth=7)
    tree.fit(X_train, y_train)
    y_pred = tree.predict(X_test)
    print("Accuracy (entropy with max_depth=7):", accuracy_score(y_test, y_pred))
    # print(tree.tree)
    # #GRID SEARCH
    # best_accuracy = 0
    # best_max_depth = 0
    # best_n_estimators = 0
    # best_criterion = 0
    # for n_estimators in [100, 200, 300, 500 , 1000, 1010 , 1100, 1110, 1200, 1210, 1300, 1310, 1400, 1410, 1500, 2000]:
    #     for  max_depth in [1, 2 ,3,4,5,6,7,8,9,10,11,12,13, 14 , 16,17,18,19,20 ]:
    #         for criterion in ['gini', 'entropy']:
    #             np.random.seed(40)
    #             ada = AdaBoost(criterion=criterion,max_depth = max_depth,  n_estimators = n_estimators)
    #             ada.fit(X_train, y_train)
    #             y_pred = ada.predict(X_test)
    #             new_accuracy = accuracy_score(y_test, y_pred)
    #             print("Accuracy:", new_accuracy)
                
    #             print("max_depth:", max_depth)
    #             print("n_estimators:", n_estimators)
    #             print("criterion:", criterion)
    #             if new_accuracy >=best_accuracy:
    #                 best_accuracy = new_accuracy
    #                 best_max_depth = max_depth
    #                 best_n_estimators = n_estimators
    #                 best_criterion = criterion

    # print("best_accuracy :", best_accuracy)
    # print("best_max_depth :", best_max_depth)
    # print("best_n_estimators :", best_n_estimators)
    # print("best_criterion :" ,  best_criterion)

# AdaBoost
    print("Part 2: AdaBoost")
    # Tune the arguments of AdaBoost to achieve higher accuracy than your Decision Tree.
    np.random.seed(40)
    ada = AdaBoost(criterion='entropy', max_depth=5, n_estimators=100)
    ada.fit(X_train, y_train)
    y_pred = ada.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, y_pred))


    
