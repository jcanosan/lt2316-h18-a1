# LT2316 H18 Assignment 1

Git project for implementing assignment 1 in [Asad Sayeed's](https://asayeed.github.io) machine learning class in the University of Gothenburg's Masters
of Language Technology programme.

The included dataset comes from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Balance+Scale).

## Your notes

The folder "other datasets" includes some very small datasets that I have been using to test my program.

Acknowledgements to the following tutorial, which has helped me a lot understanding how to apply the recursion to my train function:\
    https://www.python-course.eu/Decision_Trees.php


Revision:\
    Fixed the Prediction function so it works as intended in the lab instructions:\
        - Now the function only raises ValueError if there is no id3 model trained.\
        - If no prediction can be find in the tree, it takes the most common value of the target attribute.