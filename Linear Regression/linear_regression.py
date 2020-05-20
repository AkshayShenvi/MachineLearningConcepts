import numpy as np
import math

import sys

def convert_to_poly(training_features,degree):
    data_fitted_to_poly1=[]
    for row_object in training_features:
        rows=[]
        rows.append(1)
        for feature in row_object:
            for i in range(1,degree+1):
                rows.append(feature**i)
        data_fitted_to_poly1.append(rows)
    return np.array(data_fitted_to_poly1)


if __name__ == "__main__":
    if len(sys.argv) == 5:
        try:
            #import txt files
            training_data= np.loadtxt(str(sys.argv[1]))
            test_data= np.loadtxt(str(sys.argv[4]))

            degree = int(sys.argv[2])
            lamda= int(sys.argv[3])

            #Features of training data
            training_features=training_data[:,:-1]

            #Features of test data
            test_features=test_data[:,:-1]

            #Target of training set
            target_data=training_data[:,-1]

            #Target of test set
            target_data_test=test_data[:,-1]

            # Get polynomial with degrees for Training data
            data_fitted_to_poly=convert_to_poly(training_features,degree)

            # Get polynomials with degrees for test data
            poly_fitted_test_data=convert_to_poly(test_features,degree)

            #Calculate the Weights.
            w= np.dot(np.linalg.pinv((lamda*np.identity(data_fitted_to_poly.shape[1]))+np.dot(data_fitted_to_poly.T,data_fitted_to_poly)),  np.dot(data_fitted_to_poly.T , target_data))
            
            #Print Weights
            for index,val in enumerate(w):
                print("w%.f=%.4f"%(index,val))
            
            
            y_actual_test=[]
            for index,i in enumerate(poly_fitted_test_data):
                temp_y=np.dot(w.T,i)
                y_actual_test.append(temp_y)
                print("ID=%5d, output=%14.4f, target value = %10.4f, squared error = %.4f"%(index+1,temp_y,target_data_test[index],(temp_y-target_data_test[index])**2))
            y_actual_test=np.array(y_actual_test)
        except Exception as e:
            print(e)

    else:
        print("Invalid number of input arguments. Please check the number of input arguments once Again")
        print("Thank You!!!")
    
    