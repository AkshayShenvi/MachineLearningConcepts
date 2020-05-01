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

