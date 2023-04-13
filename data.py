import torch
import pandas as pd
import numpy as np
import sklearn

# read from csv into pd dataframe
X_df = pd.read_csv('/Users/emmachen/Documents/aqlab/uniprot-nn/data/uniprot-X-aa-count.csv') 
y_df = pd.read_csv('/Users/emmachen/Documents/aqlab/uniprot-nn/data/uniprot-y-label-encoded.csv')

# convert to numpy array
X = np.array(X_df.values,'float') 
y = np.array(y_df.values,'float')

# train test split
from sklearn.model_selection import train_test_split
X_train, X_valtest, y_train, y_valtest = train_test_split(X, y, test_size=0.3, random_state=42)
X_test, X_val, y_test, y_val = train_test_split(X, y, test_size=0.5, random_state=42)

# convert to torch tensor
xtrain = torch.tensor(X_train).type(torch.float32) 
ytrain = torch.tensor(y_train).type(torch.long)
xval = torch.Tensor(X_val).type(torch.float32)
yval = torch.Tensor(y_val).type(torch.long)
xtest = torch.Tensor(X_test).type(torch.float32)
ytest = torch.Tensor(y_test).type(torch.long)


# torch datasets
train_dataset = torch.utils.data.TensorDataset(xtrain,ytrain) 
val_dataset = torch.utils.data.TensorDataset(xval,yval)
test_dataset = torch.utils.data.TensorDataset(xtest,ytest)

