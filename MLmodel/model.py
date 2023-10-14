# Frist Create Custom Train Test Splitting 
import numpy as np
import pandas as pd

def split_train_test(data,test_ratio):
    np.random.seed(42) 
    shuffled=np.random.permutation(len(data))
    test_set_size=int(len(data)*test_ratio)
    test_indices=shuffled[:test_set_size]
    train_indices=shuffled[test_set_size:]
    return data.iloc[train_indices],data.iloc[test_indices]


# train_set,test_set=split_train_test(housing,0.2)
columns = [
    "CRIM",
    "ZN",
    "INDUS",
    "CHAS",
    "NOX",
    "RM",
    "AGE",
    "DIS",
    "RAD",
    "TAX",
    "PTRATIO",
    "B",
    "LSTAT",
    "PRICE"
]
data=pd.read_csv('housing.csv')
data.columns=columns

train_set,test_set=split_train_test(data,0.2)
# print(train_set)
# print(data.size)
# print(train_set.size+test_set.size) Checking if my custom TrainTest method is working or not

#Custom Linear Regression for House Prediction
class CustomLinearRegression:
    def __init__(self,learningRate,epoch):
        self.learningRate=learningRate
        self.epoch=epoch
    
    # Training Function
    def fit(self,X,y):
        self.X=X
        self.y=y.reshape(-1,1) 
        self.coef_=np.random.random() #Random Value for the Slop
        self.intercept_=np.random.random() #Random Value for the Intercept
        for i in range(self.epoch):
            error_interCept=-2*np.sum(y-self.coef_*self.X+self.intercept_)
            error_slop=-2*np.sum((y-self.coef_*self.X+self.intercept_)*self.X)
            
            self.intercept_=self.intercept_-(self.learningRate*error_interCept)
            self.coef_=self.coef_-(self.learningRate*error_slop)

    
model=CustomLinearRegression(0.001,50)
print(train_set)
        
            
