# Bayes Classifier for continuous Values

Here I will be showing how to model a Bayes Classifier for continuous data and from scratch in Python.

*Note: I will be using Numpy and pandas libraries for handelling data*

Importing the libraries that we need for building the classifier.

```python

import pandas as pd
import numpy as np
import math
import sys

```

Next we need to read the training and test data from the CSV. 

*We would be using pendigits_training.txt and pendigits_test.txt for training and testing the data. The model we would be building would work on all of the datasets provided in the repo.*

We would be using **pd.read_csv** with **"\s+"** as the seperator as our datasets are space seperated.
```python
data= pd.read_csv(str(sys.argv[1]),sep="\s+",header=None)
data_test=pd.read_csv(str(sys.argv[2]),sep="\s+",header=None)

```

To have a general view of the data (i.e training data).
```python
data.head(10)
```
![Image of 10 rows of training data](https://github.com/AkshayShenvi/MachineLearningConcepts/blob/master/Bayes%20Classifier/Images/data_example.PNG)

We can see through the above image that there are 17 columns (i.e 0-16) where the columns 0-15 are the features or inputs, while the last column is labels or target.

Another way of knowing your data in general is printing out the shape of the dataset.

```python
data.shape
```
![Image of Shape of the dataset](https://github.com/AkshayShenvi/MachineLearningConcepts/blob/master/Bayes%20Classifier/Images/data_shape.PNG)

Here as we can see here the dataset consists of 7494 rows (data objects) and 17 columns of which the last column is the target column.

Next, we want to separate out the features from the target, so we take the index of the target.

```python
index_of_label=(data.shape[1])-1
```
And then the index of the features.
```python
index_of_features=(data.shape[1])-2
```

