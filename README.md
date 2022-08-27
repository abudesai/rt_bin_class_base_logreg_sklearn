Logistic Regression Classifier in SciKitLearn for Binary Classification - Base problem category as per Ready Tensor specifications.

* logistic regression
* sklearn
* python
* pandas
* numpy
* scikit-optimize
* flask
* nginx
* uvicorn
* docker
* binary classification

This is a Binary Classifier that uses Logistic Regression implemented through SciKitLearn.

The classifier starts by fitting a logistic function to the explanatory variables (features) and the dependent variable (label) to detemrmine the probability of a group of features belonging to a class. 

The data preprocessing step includes missing data imputation, standardization, one-hot encoding for categorical variables, datatype casting, etc. The missing categorical values are imputed using the most frequent value if they are rare. Otherwise if the missing value is frequent, they are give a "missing" label instead. Missing numerical values are imputed using the mean and a binary column is added to show a 'missing' indicator for the missing values. Numerical values are also scaled using a Yeo-Johnson transformation in order to get the data close to a Gaussian distribution. 

Hyperparameter Tuning (HPT) is conducted by finding the optimal penalty method for the model, regularization strength, and l1 ratio if the penalty method is chosen to be elastic net.

During the model development process, the algorithm was trained and evaluated on a variety of datasets such as email spam detection, customer churn, credit card fraud detection, cancer diagnosis, and titanic passanger survivor prediction.

This Binary Classifier is written using Python as its programming language. Scikitlearn is used to implement the main algorithm, create the data preprocessing pipeline,  and evaluate the model. Numpy, pandas, and feature_engine are used for the data preprocessing steps. SciKit-Optimize was used to handle the HPT. Flask + Nginx + gunicorn are used to provide web service which includes two endpoints- /ping for health check and /infer for predictions in real time.


