# Databricks notebook source
from sklearn import datasets, svm
import mlflow.sklearn
mlflow.sklearn.autolog()

# COMMAND ----------

data = datasets.load_iris()
model = svm.SVC(C=2)
model.fit(data.data, data.target)