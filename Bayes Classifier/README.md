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

Next, we would need the total number of data objects.
```python
total_labels=data[index_of_label].count()
```

We would also need to take out the unique classes, count of each class, and frequency of each class.
```python
#Unique Classes
unique_classes=sorted(data[index_of_label].unique())

# Frequency of each class
label_probability=data.iloc[:,-1].value_counts()

#Per Class Frequency
per_class_probability=label_probability/total_labels
```

As in this Bayes Classifier, we will be assuming that the data is of Gaussian Distribution (i.e. we will use Gaussian as our basis function). You can replace this by your own basis function.

For Gaussian Distribution, we would need to calculate the mean and the variance of each feature and their classes.

```python
# Data mean
data_means = data.groupby(index_of_label).mean()

# Data variance
data_variance = data.groupby(index_of_label).var()
```
Our mean and variances would look like this:

**Mean**
![Image of mean](https://github.com/AkshayShenvi/MachineLearningConcepts/blob/master/Bayes%20Classifier/Images/means.PNG)

**Variance**
![Image of variance](https://github.com/AkshayShenvi/MachineLearningConcepts/blob/master/Bayes%20Classifier/Images/var.PNG)

Next, we will limit our variance to 0.0001(i.e. If the variance is less than 0.0001 we would replace it with 0.0001). This would help us to avoid a variance with 0 as its value.
```python
#Limit Variance to 0.0001 i.e (SD to 0.01)
corr_var=data_variance
corr_var[corr_var.iloc[:,:] < 0.0001]= 0.0001
```

Next we would calculate the likelihood. i.e for all the test data points, and for each dimension.
```python
# Calculate likelihood 
prob=[]
prob1=[]
for data_rows in range(len(data_test)):
    prob=[]
    index_of_mean=0
    for class_num in unique_classes:
        p=pofXgivenY(data_test.iloc[data_rows,:-1],data_means.iloc[index_of_mean,:],corr_var.iloc[index_of_mean,:])

        prob.append(per_class_probability[class_num]*np.prod(p))

        index_of_mean+=1
    prob1.append(prob)
```

Now that we have calculated the likelihood for each test data point for each dimension, we will calculate the accuracy and print out the other parameters.
```python
# Calculate and check for multiple maximums
final_length_array=[]
for class_object in range(len(prob1)):
    maximum=0
    index_for_max_array=[]
    i_max=np.argmax(prob1[class_object])
    maximum=prob1[class_object][i_max]
    for index in range(len(prob1[class_object])):
        if prob1[class_object][index] == maximum:
            index_for_max_array.append(prob1[class_object][index])
    final_length_array.append(len(index_for_max_array))




# Print Test parameters i.e.(ID,predicted,probability,true,accuracy)

accuracy=0
for class_object in range(len(prob1)):
    temp=np.argmax(prob1[class_object])
    if unique_classes[temp] == data_test.iloc[class_object,data_test.shape[1]-1]:
        accuracy+=1


    else:
        if final_length_array[class_object]>1:
            accuracy+=(1/final_length_array[class_object])
    i_max=np.argmax(prob1[class_object])
    print("ID=%5d, predicted=%3d, probability = %.4f, true=%3d, accuracy=%4.2f\n"%(class_object+1, unique_classes[i_max], (np.amax(prob1[class_object])/np.sum(prob1[class_object])), (data_test.iloc[class_object,data_test.shape[1]-1]), (accuracy/len(prob1))*100))
# Final Accuracy
print("classification accuracy=%6.4f"%((accuracy/len(prob1))*100))
```
