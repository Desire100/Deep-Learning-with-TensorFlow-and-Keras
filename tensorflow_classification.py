# TensorFlow Classification 

import pandas as pd

diabetes = pd.read_csv('pima-indians-diabetes.csv')

diabetes.head()
diabetes.columns

# Clean the Data

cols_to_norm = ['Number_pregnant', 'Glucose_concentration', 'Blood_pressure', 'Triceps',
       'Insulin', 'BMI', 'Pedigree']

diabetes[cols_to_norm] = diabetes[cols_to_norm].apply(lambda x: (x - x.min()) / (x.max() - x.min()))
diabetes.head()

# Feature Columns
diabetes.columns

import tensorflow as tf
"""
Continuous Features
Number of times pregnant
Plasma glucose ,concentration a 2 hours in an oral glucose tolerance test,
Diastolic blood pressure (mm Hg),
Triceps skin fold thickness (mm),
2-Hour serum insulin (mu U/ml),
Body mass index (weight in kg/(height in m)^2),
Diabetes pedigree function
"""

num_preg = tf.feature_column.numeric_column('Number_pregnant')
plasma_gluc = tf.feature_column.numeric_column('Glucose_concentration')
dias_press = tf.feature_column.numeric_column('Blood_pressure')
tricep = tf.feature_column.numeric_column('Triceps')
insulin = tf.feature_column.numeric_column('Insulin')
bmi = tf.feature_column.numeric_column('BMI')
diabetes_pedigree = tf.feature_column.numeric_column('Pedigree')
age = tf.feature_column.numeric_column('Age')

## Categorical Features 

"""
If you know the set of all possible feature values of a column and
 there are only a few of them, you can use 
 categorical_column_with_vocabulary_list. If you don't know 
 the set of possible values in advance you can use 
 categorical_column_with_hash_bucket
"""

assigned_group = tf.feature_column.categorical_column_with_vocabulary_list('Group',['A','B','C','D'])
# Alternative
# assigned_group = tf.feature_column.categorical_column_with_hash_bucket('Group', hash_bucket_size=10)

# Converting Continuous to Categorical

import matplotlib.pyplot as plt

diabetes['Age'].hist(bins=20)
age_buckets = tf.feature_column.bucketized_column(age, boundaries=[20,30,40,50,60,70,80])

# Putting them together
feat_cols = [num_preg ,plasma_gluc,dias_press ,tricep ,insulin,bmi,diabetes_pedigree ,assigned_group, age_buckets]

# Train Test Split
diabetes.head()
diabetes.info()
x_data = diabetes.drop('Class',axis=1)
labels = diabetes['Class']

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x_data,labels,test_size=0.33, random_state=101)

# Input Function
input_func = tf.estimator.inputs.pandas_input_fn(x=X_train,y=y_train,batch_size=10,num_epochs=1000,shuffle=True)

# Creating the Model
model = tf.estimator.LinearClassifier(feature_columns=feat_cols,n_classes=2)
model.train(input_fn=input_func,steps=1000)

# Useful link ofr your own data
# https://stackoverflow.com/questions/44664285/what-are-the-contraints-for-tensorflow-scope-names

# Evaluation

eval_input_func = tf.estimator.inputs.pandas_input_fn(
      x=X_test,
      y=y_test,
      batch_size=10,
      num_epochs=1,
      shuffle=False)

results = model.evaluate(eval_input_func)

results

# Predictions

pred_input_func = tf.estimator.inputs.pandas_input_fn(
      x=X_test,
      batch_size=10,
      num_epochs=1,
      shuffle=False)

# Predictions is a generator! 
predictions = model.predict(pred_input_func)

list(predictions)


# DNN Classifier

dnn_model = tf.estimator.DNNClassifier(hidden_units=[10,10,10],feature_columns=feat_cols,n_classes=2)

# UH OH! AN ERROR. Check out the video to see why and how to fix.
# https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/feature_column/feature_column.py

dnn_model.train(input_fn=input_func,steps=1000)

embedded_group_column = tf.feature_column.embedding_column(assigned_group, dimension=4)
input_func = tf.estimator.inputs.pandas_input_fn(x=X_train,y=y_train,batch_size=10,num_epochs=1000,shuffle=True)
dnn_model = tf.estimator.DNNClassifier(hidden_units=[10,10,10],feature_columns=feat_cols,n_classes=2)

dnn_model.train(input_fn=input_func,steps=1000)

eval_input_func = tf.estimator.inputs.pandas_input_fn(
      x=X_test,
      y=y_test,
      batch_size=10,
      num_epochs=1,
      shuffle=False)

dnn_model.evaluate(eval_input_func)

