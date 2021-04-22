
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 28 22:17:55 2021

@author: ayomy
"""

import numpy as np

class PerceptronPolyKernel:
    
    def __init__(self,a,b,d,max_iter):
        d > 0; a >= 0; b> 0;
        self.alphas_1 = np.zeros(len(X_train)) #initialize alphas to zero
        self.bias_1 = 0
        self.alphas_2 = np.zeros(len(X_train)) #initialize alphas to zero
        self.bias_2 = 0
        self.alphas_3 = np.zeros(len(X_train)) #initialize alphas to zero
        self.bias_3 = 0
        self.alphas_4 = np.zeros(len(X_train)) #initialize alphas to zero
        self.bias_4 = 0
        self.a = a
        self.b = b
        self.d = int(d)
        self.max_iter = max_iter
        self.X_train_f = X_train
        

    
# polynomial_kernel function      Thsi does the computation for an instance
    def poly_kernel(self,X_train,i):
        Kernel_matrix = (self.a + self.b*(np.dot(self.X_train_f,X_train[i])))**self.d
        return Kernel_matrix


# Kernelized perceptron training
    def fit1(self,X_train,y_train_1,learn_rate1):
        y_train_1 = [1 if b == 1 else -1 for b in y_train_1]
        for t in range(self.max_iter):
            for i in range(len(X_train)):
                act = np.dot(self.alphas_1,self.poly_kernel(X_train,i)) + self.bias_1  # Summation of all the instance
                if np.sign(act) == 1:
                    y_predicted = 1
                else:
                    y_predicted = -1
                    
                if y_train_1[i] != y_predicted :
                    self.alphas_1[i] = self.alphas_1[i] + learn_rate1*y_train_1[i]
                    self.bias_1 = self.bias_1 + learn_rate1* y_train_1[i]
                    
                
        return self.alphas_1,self.bias_1
    
    def fit2(self,X_train,y_train_2,learn_rate2): 
       y_train_2 = [1 if i == 1 else -1 for i in y_train_2] 
       for t in range(self.max_iter):
           for i in range(len(X_train)):
               act = np.dot(self.alphas_2,self.poly_kernel(X_train,i)) + self.bias_2  # Summation of all the instance
               if np.sign(act) == 1:
                   y_predicted = 1
               else:
                   y_predicted = -1
                   
               if y_train_2[i] != y_predicted :
                   self.alphas_2[i] = self.alphas_2[i] + learn_rate2*y_train_2[i]
                   self.bias = self.bias_2 + learn_rate2* y_train_2[i]
                   
               
       return self.alphas_2,self.bias_2

    def fit3(self,X_train,y_train_3,learn_rate3): 
        y_train_3 = [1 if i == 1 else -1 for i in y_train_3]  
        for t in range(self.max_iter):
            for i in range(len(X_train)):
                act = np.dot(self.alphas_3,self.poly_kernel(X_train,i)) + self.bias_3  # Summation of all the instance
                if np.sign(act) == 1:
                    y_predicted = 1
                else:
                    y_predicted = -1
                    
                if y_train_3[i] != y_predicted :
                    self.alphas_3[i] = self.alphas_3[i] + learn_rate3*y_train_3[i]
                    self.bias_3 = self.bias_3 + learn_rate3* y_train_3[i]
                    
                
        return self.alphas_3,self.bias_3          
            
    def fit4(self,X_train,y_train_4,learn_rate4):
        y_train_4 = [1 if i == 1 else -1 for i in y_train_4]
        for t in range(self.max_iter):
            for i in range(len(X_train)):
                act = np.dot(self.alphas_4,self.poly_kernel(X_train,i)) + self.bias_4  # Summation of all the instance
                if np.sign(act) == 1:
                    y_predicted = 1
                else:
                    y_predicted = -1
                    
                if y_train_4[i] != y_predicted :
                    self.alphas_4[i] = self.alphas_4[i] + learn_rate4*y_train_4[i]
                    self.bias_4 = self.bias_4 + learn_rate4* y_train_4[i]
                    
                
        return self.alphas_4,self.bias_4
           
     # This function predict all the final label   
    def predict1(self,X_test): 
       y_predict_final = np.zeros(len(X_test)) 
       for l in range(len(X_test)):
            act = np.dot(self.alphas_1,self.poly_kernel(X_test,l)) + self.bias_1
            if np.sign(act) == 1:
               y_predict_final[l] = 1
            else:
               y_predict_final[l] = -1
           
       return y_predict_final
   
     # This function predict all the final label   
    def predict2(self,X_test): 
       y_predict_final = np.zeros(len(X_test)) 
       for l in range(len(X_test)):
            act = np.dot(self.alphas_2,self.poly_kernel(X_test,l)) + self.bias_2
            if np.sign(act) == 1:
               y_predict_final[l] = 1
            else:
               y_predict_final[l] = -1
           
       return y_predict_final
   
     # This function predict all the final label   
    def predict3(self,X_test): 
       y_predict_final = np.zeros(len(X_test)) 
       for l in range(len(X_test)):
            act = np.dot(self.alphas_3,self.poly_kernel(X_test,l)) + self.bias_3
            if np.sign(act) == 1:
               y_predict_final[l] = 1
            else:
               y_predict_final[l] = -1
           
       return y_predict_final
   
     # This function predict all the final label   
    def predict4(self,X_test): 
       y_predict_final = np.zeros(len(X_test)) 
       for l in range(len(X_test)):
            act = np.dot(self.alphas_4,self.poly_kernel(X_test,l)) + self.bias_4
            if np.sign(act) == 1:
               y_predict_final[l] = 1
            else:
               y_predict_final[l] = -1
           
       return y_predict_final
   
    def predict_bin_label_1(self):
        y_predict_bin_1 = [1 if i == 1 else 0 for i in self.predict1(X_test)]
        return y_predict_bin_1
    
    def predict_bin_label_2(self):
        y_predict_bin_2 = [1 if i == 1 else 0 for i in self.predict2(X_test)]
        return y_predict_bin_2
    
    def predict_bin_label_3(self):
        y_predict_bin_3 = [1 if i == 1 else 0 for i in self.predict3(X_test)]
        return y_predict_bin_3
    
    def predict_bin_label_4(self):
        y_predict_bin_4 = [1 if i == 1 else 0 for i in self.predict4(X_test)]
        return y_predict_bin_4
    
    def prediction_accuracy1(self,y_test_1):
        correctcount = 0
        wrongcount = 0
        y_predict_final1 = self.predict_bin_label_1()
        testlabel_and_predictedlabel = list(zip(y_test_1,y_predict_final1))
        for i in range(len(testlabel_and_predictedlabel)):
            if (testlabel_and_predictedlabel[i][0]) == (testlabel_and_predictedlabel[i][1]):
                correctcount += 1
            else:
                wrongcount += 1
        accuracyratio = (correctcount/(correctcount+wrongcount))
        return accuracyratio 
    
    def prediction_accuracy2(self,y_test_2):
        correctcount = 0
        wrongcount = 0
        y_predict_final2 = self.predict_bin_label_2()
        testlabel_and_predictedlabel = list(zip(y_test_1,y_predict_final2))
        for i in range(len(testlabel_and_predictedlabel)):
            if (testlabel_and_predictedlabel[i][0]) == (testlabel_and_predictedlabel[i][1]):
                correctcount += 1
            else:
                wrongcount += 1
        accuracyratio = (correctcount/(correctcount+wrongcount))
        return accuracyratio 
    
    def prediction_accuracy3(self,y_test_3):
        correctcount = 0
        wrongcount = 0
        y_predict_final3 = self.predict_bin_label_3()
        testlabel_and_predictedlabel = list(zip(y_test_1,y_predict_final3))
        for i in range(len(testlabel_and_predictedlabel)):
            if (testlabel_and_predictedlabel[i][0]) == (testlabel_and_predictedlabel[i][1]):
                correctcount += 1
            else:
                wrongcount += 1
        accuracyratio = (correctcount/(correctcount+wrongcount))
        return accuracyratio 
    
    def prediction_accuracy4(self,y_test_4):
        correctcount = 0
        wrongcount = 0
        y_predict_final4 = self.predict_bin_label_4()
        testlabel_and_predictedlabel = list(zip(y_test_1,y_predict_final4))
        for i in range(len(testlabel_and_predictedlabel)):
            if (testlabel_and_predictedlabel[i][0]) == (testlabel_and_predictedlabel[i][1]):
                correctcount += 1
            else:
                wrongcount += 1
        accuracyratio = (correctcount/(correctcount+wrongcount))
        return accuracyratio 
    
    #encoding part
    def encode_part1(self):
        final_y = np.vstack([self.predict_bin_label_1(),self.predict_bin_label_2(),self.predict_bin_label_3(),self.predict_bin_label_4()]) 
        final_y = np.transpose(final_y)
        final_y = final_y.tolist()
        final_y_predict = []
        for i in final_y: 
             if i == [0,0,0,0]:
                final_y_predict.append(0)
             elif i == [0,0,0,1]:
                final_y_predict.append(1)
             elif i == [0,0,1,0]:
                final_y_predict.append(2)
             elif i == [0,0,1,1]:
                final_y_predict.append(3)
             elif i == [0,1,0,0]:
                final_y_predict.append(4)
             elif i == [0,1,0,1]:
                final_y_predict.append(5) 
             elif i == [0,1,1,0]:
                final_y_predict.append(6)
             elif i == [0,1,1,1]:
                final_y_predict.append(7)
             elif i == [1,0,0,0]:
                final_y_predict.append(8)
             else:
                final_y_predict.append(9)
        return final_y_predict
        
    def encode_part2(self,y_test):
        final_y_test = y_test.tolist()
        final_y_test_a = []
        for i in final_y_test: 
             if i == [0,0,0,0]:
                final_y_test_a.append(0)
             elif i == [0,0,0,1]:
                final_y_test_a.append(1)
             elif i == [0,0,1,0]:
                final_y_test_a.append(2)
             elif i == [0,0,1,1]:
                final_y_test_a.append(3)
             elif i == [0,1,0,0]:
                final_y_test_a.append(4)
             elif i == [0,1,0,1]:
                final_y_test_a.append(5) 
             elif i == [0,1,1,0]:
                final_y_test_a.append(6)
             elif i == [0,1,1,1]:
                final_y_test_a.append(7)
             elif i == [1,0,0,0]:
                final_y_test_a.append(8)
             elif i == [1,0,0,1]:
                final_y_test_a.append(9)
        return final_y_test_a
    
    def final_prediction_accuracy(self):
        correctcount = 0
        wrongcount = 0
        y_predict_final = self.encode_part1()
        testlabel_and_predictedlabel = list(zip(self.encode_part2(y_test),y_predict_final))
        for i in range(len(testlabel_and_predictedlabel)):
            if (testlabel_and_predictedlabel[i][0]) == (testlabel_and_predictedlabel[i][1]):
                correctcount += 1
            else:
                wrongcount += 1
        accuracyratio = (correctcount/(correctcount+wrongcount))
        return accuracyratio 
    
    

np.random.seed (0)
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_svmlight_file
from joblib import Memory
mem = Memory("./mycache")

@mem.cache
def get_data():
    data = load_svmlight_file("mnist.scale.bz2")
    return data[0],data[1]

X,y = get_data()
y= y.astype(int)
y = y.tolist()
new_y = []
for i in y:
    if i == 0:
        new_y.append([0,0,0,0])
    elif i == 1:
        new_y.append([0,0,0,1])
    elif i == 2:
        new_y.append([0,0,1,0])
    elif i == 3:
        new_y.append([0,0,1,1])
    elif i == 4:
        new_y.append([0,1,0,0])
    elif i == 5:
        new_y.append([0,1,0,1]) 
    elif i == 6:
        new_y.append([0,1,1,0])
    elif i == 7:
        new_y.append([0,1,1,1])
    elif i == 8:
        new_y.append([1,0,0,0])
    else:
        new_y.append([1,0,0,1])
new_y = np.array(new_y)
X_train,X_test,y_train,y_test= train_test_split(X.toarray(),new_y,test_size = 0.3)

y_train_1 = y_train[:,0] 
y_train_2 = y_train[:,1]  
y_train_3 = y_train[:,2]
y_train_4 = y_train[:,3]

y_test_1 = y_test[:,0] 
y_test_2 = y_test[:,1]  
y_test_3 = y_test[:,2]
y_test_4 = y_test[:,3]
    
bin = PerceptronPolyKernel(1,1,8,10)
bin.fit1(X_train,y_train_1,0.1)
bin.fit2(X_train,y_train_2,0.1)
bin.fit3(X_train,y_train_3,0.1)
bin.fit4(X_train,y_train_4,0.1)

print("The kernelized perceptron ECOC algorithm performance accuracy is " + str(bin.final_prediction_accuracy()))