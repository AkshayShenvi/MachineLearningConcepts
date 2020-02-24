# -*- coding: utf-8 -*-

"""
Created on Fri Feb  7 22:00:01 2020

@author: Akshay Shenvi
"""

#import libraries
import pandas as pd
import numpy as np
import math
import sys

#run file with the following command:  
#python (file name) (training data in text file) (test data in a text file)
#eg.
#%python naive_bayes.py pendigits_training.txt pendigits_test.txt


# Gaussian Formula to find liklihood
def pofXgivenY(X,mean,variance):
    
    p=1/(np.sqrt(2*np.pi*variance))*np.exp((-(X-mean)**2)/(2*variance))
    
    return p

if __name__ == "__main__":
    
    if len(sys.argv) == 3:
        try:
            data= pd.read_csv(str(sys.argv[1]),sep="\s+",header=None)
            data_test=pd.read_csv(str(sys.argv[2]),sep="\s+",header=None)
            
            #Index of Label
            index_of_label=(data.shape[1])-1

            #Index of Features
            index_of_features=(data.shape[1])-2

            #Total number of Labels
            total_labels=data[index_of_label].count()
            
            #Unique Classes
            unique_classes=sorted(data[index_of_label].unique())
            
            # Frequency of each class
            
            label_probability=data.iloc[:,-1].value_counts()


            #Per Class Frequency
            
            per_class_probability=label_probability/total_labels
            
            # Data mean
            data_means = data.groupby(index_of_label).mean()

            # Data variance
            data_variance = data.groupby(index_of_label).var()
            

            #Limit Variance to 0.0001 i.e (SD to 0.01)
            corr_var=data_variance
            corr_var[corr_var.iloc[:,:] < 0.0001]= 0.0001
            

            #Print Class Attribute Mean and Standard Diviation
            for i in unique_classes:
                for j in range(index_of_features):
                    print("Class %d, attribute %d, mean = %.2f, std = %.2f"% (i,j+1,data_means[j][i],math.sqrt(corr_var[j][i])))


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
        except Exception as e:
            print(e)

        


    else:
        print("Invalid number of input arguments. Please check the number of input arguments once Again")
        print("Thank You!!!")