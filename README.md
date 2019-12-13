# AI Engine for Heart Disease Prediction

_Written in **Python3**. Using **Scikit-learn/Keras** ML framework._

## Data
- “Heart disease,” Mayo Clinic, 22-Mar-2018. [Online]. Available: https://www.mayoclinic.org/diseases-conditions/heart-disease/symptoms-causes/syc-20353118. [Accessed: 30-Nov-2019].
- A. Janosi, W. Steinbrunn, M. Pfisterer, and R. Detrano, “ Heart Disease Data Set,” UCI Machine Learning Repository.
- Input : 13 features
- Output : 0 (no heart disease) or 1 (have heart disease)

## Learning Algorithms / Models
- Perceptron
- Logistic Regression
- KNN
- Kernelized SVM
- Desision Tree
- Random Forest
- DNN

## Approaches
- Start from the linear model : Perceptron
- Analyze the features via the weights of Logistic Regression
  * Show that the dataset has bias and demonstrates the limitation of linear models
- Continue to Non Linear Models : KNN / Kernelized SVM / Desision Tree / Random Forest / DNN
  * Show that non-linear models can learn a better expression of the model and get better performance
- Test the generalization of model with an another unseen dataset
  * Drop the last 3 features because most of the values of the last 3 features in the unseen dataset are unknown.
- Metrics : Accuracy / Precision / Recall / F1 Score

## Commands

- Get the experiment results of Perceptron 

```
python3 perceptron_heart.py
```

- Get the experiment results of KNN 

```
python3 KNN.py
```

- Get the experiment results of Kernel SVM

```
python3 kernelSVM.py
```

- Get the experiment results of Decision Tree

```
python3 decisionTree.py
```

- Get the experiment results of Random Forest Model

```
python3 randomForest.py
```

- Get the experiment results of Logistic Regression Model 

```
python3 LR.py
```

- Get the experiment results of Neural Network Model

```
python3 NN.py
```

## Results
- Here is the link to the CodaLab worksheet, which demonstrates our executing results:
  * https://worksheets.codalab.org/worksheets/0x62e448be3d0a4c44aca4027cc700d139
