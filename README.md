# Automated_ML
This code only needs a dataset as input and will output the best possible supervised learning model possible
After downloading the model
For Regression
```
from pycaret.regression import load_model
pipeline=load_model('trained_model')
```
After downloading the model
For Classification
```
from pycaret.classification import load_model
pipeline=load_model('trained_model')
```
