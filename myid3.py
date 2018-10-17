# Module file for implementation of ID3 algorithm.

# You can add optional keyword parameters to anything, but the original
# interface must work with the original test file.
# You will of course remove the "pass".

import os, sys
import numpy as np
import pandas as pd
import sklearn.metrics as metr
import dill
# You can add any other imports you need.


class DecisionTree:
    def __init__(self, load_from=None):
        # Fill in any initialization information you might need.
        #
        # If load_from isn't None, then it should be a file *object*,
        # not necessarily a name. (For example, it has been created with
        # open().)
        print("Initializing classifier.")
        if load_from is not None:
            print("Loading from file object.")
            self.id3_tree = dill.load(load_from)

    def train(self, X, y, attrs, prune=False):
        # Doesn't return anything but rather trains a model via ID3
        # and stores the model result in the instance.
        # X is the training data, y are the corresponding classes the
        # same way "fit" worked on SVC classifier in scikit-learn.
        # attrs represents the attribute names in columns order in X.
        #
        # Implementing pruning is a bonus question, to be tested by
        # setting prune=True.
        #
        # Another bonus question is continuously-valued data. If you try this
        # you will need to modify predict and test.

        # If all the values in the "class" column are the same, return that
        # value
        if len(y.unique()) <= 1:
            return(y.unique()[0])

        # If number of predicting attributes is empty, then Return the single
        # node tree Root, with label = most common value of the target attribute
        # in the examples.
        elif len(attrs) == 0:
            return y.mode()[0]

        else:
            # Entropy of the "class"
            hy = DecisionTree.entropy(self, y.value_counts(normalize=True))
            ig_attrs = dict()
            y_index = dict()
            for attr in attrs:
                # Obtain the indexes of the unique values of one attr and
                # mapping them to the "class"
                X_index_attr = dict()
                y_index_attr = dict()
                for v in X[attr].unique():
                    X_index_attr[v] = \
                        [i for i in X[attr].index[X[attr] == v].tolist()]
                    y_index_attr[v] = y[X_index_attr[v]]
                # Storing the mappings of "class" to build the tree
                y_index[attr] = y_index_attr

                # Calculate the IG and store it inside a dictionary
                ig_attr = DecisionTree.information_gain(self, hy, y_index_attr,
                                                        X, attr)
                ig_attrs[attr] = ig_attr

            # Choose the attribute with the max IG
            max_ig_attr = \
                sorted(ig_attrs.items(), reverse=True, key=lambda x: x[1])[0]

            # Create tree structure
            id3_tree = {max_ig_attr[0]:{}}

            # Remove attribute with best IG
            new_attrs = [attr for attr in attrs if attr != max_ig_attr[0]]

            # Loop inside the different values of the attribute with highest IG
            # to draw the branches
            for i in y_index[max_ig_attr[0]].keys():
                new_X = X.where(X[max_ig_attr[0]] == i).dropna()
                new_y = y.where(X[max_ig_attr[0]] == i).dropna()
                # Apply recursion
                id3_node_tree = DecisionTree.train(self, new_X, new_y, new_attrs)
                id3_tree[max_ig_attr[0]][i] = id3_node_tree

            # Update the id3_tree instance in the object
            self.id3_tree = id3_tree
            return self.id3_tree

    def predict(self, instance, id3_model, default_value):
        # Returns the class of a given instance.
        # Raise a ValueError if the class is not trained.
        """
        Predict

        Iterates through the model nested dictionary applying recursion until
        it finds a terminal node that matches the values of the attributes of
        the variable "instance".
        If no prediction can be found, returns the default_value.
        If there is no trained id3 tree, raise a ValueError.

        Args:
            Instance: it's the row in the testing data that we are trying to
            predict.
            id3_model: it's the tree model, or one of its subtrees when applying
            recursion.
            default_value: most common value of the target attribute
        """

        if id3_model is None:
            raise ValueError('The class is not trained!')
        else:
            prediction = None
            for key, value in id3_model.items():
                for attr, val in instance.items():
                    if attr == key or val == key:
                        if isinstance(value, dict):
                            prediction = DecisionTree.predict(self, instance,
                                                              value, default_value)
                        else:
                            return value
                    else:
                        pass
            if prediction is not None:
                return prediction
            else:
                # Return the most common value of target attribute.
                return default_value

    def test(self, X, y, display=False):
        # Returns a dictionary containing test statistics:
        # accuracy, recall, precision, F1-measure, and a confusion matrix.
        # If display=True, print the information to the console.
        # Raise a ValueError if the class is not trained.

        # Iterate through the different instances in the test_X dataframe and
        # call the "predict" function to predict the class
        predictions_y = list()
        for i in X.iterrows():
            predictions_y.append(DecisionTree.predict(self, i[1], self.id3_tree,
                                                      y.mode()[0]))
        predictions_y = pd.Series(predictions_y)

        # Precision, recall, accuracy, F1-measure and confusion matrix.
        precision = metr.precision_score(y, predictions_y, average=None)
        recall = metr.recall_score(y, predictions_y, average=None)
        accuracy = metr.accuracy_score(y, predictions_y)
        f1_score = metr.f1_score(y, predictions_y, average=None)
        confusion = metr.confusion_matrix(y, predictions_y)

        result = {'precision': precision,
                  'recall': recall,
                  'accuracy': accuracy,
                  'F1': f1_score,
                  'confusion-matrix': confusion}
        if display:
            for k,v in result.items():
                print("{0}: {1}".format(k, v))
        return result

    def __str__(self):
        # Returns a readable string representation of the trained
        # decision tree or "ID3 untrained" if the model is not trained.
        if isinstance(self.id3_tree, dict):
            return str(self.id3_tree)
        else:
            return "ID3 untrained"


    def save(self, output):
        # 'output' is a file *object* (NOT necessarily a filename)
        # to which you will save the model in a manner that it can be
        # loaded into a new DecisionTree instance.
        dill.dump(self.id3_tree, output)

    def entropy(self, attr_counts):
        """Entropy

        Calculates and returns the entropy of an attribute of a dataframe.

        Args:
            attr_counts: the normalized counts of that attribute in the
            y dataframe.
        """
        h = -sum(c * np.log2(c) for c in attr_counts)
        return h

    def cond_entropy(self, X_weights, y_index):
        """Conditional entropy

        For all the possible values of an attribute, calls the "entropy"
        function and sums all the resulting weighted conditional entropies.

        Args:
            X_weights: normalized weights of an attribute of the X dataframe.
            y_index: indexes in the y dataframe of all the unique values of the
            attribute in X_weights.
        """
        h_cond = 0
        for i in y_index:
            y_counts = y_index[i].value_counts(normalize=True)
            h_cond_i = X_weights[i] * DecisionTree.entropy(self, y_counts)
            h_cond += h_cond_i
        return h_cond

    def information_gain(self, h, y_index, X, attr):
        """Information Gain

        Calculates the information gain with the results of the entropy of the
        whole y dataframe and the conditional entropy of one of the attributes
        in X.

        Args:
            h: the entropy of the whole y dataframe.
            y_index: indexes in the y dataframe of all the unique values of the
            attribute in X_weights.
            X: the same X dataframe in the function "train".
            attr: the selected attribute to calculate the information gain.
        """
        X_weights = X[attr].value_counts(normalize=True)
        h_cond = DecisionTree.cond_entropy(self, X_weights, y_index)
        ig = h - h_cond
        return ig
