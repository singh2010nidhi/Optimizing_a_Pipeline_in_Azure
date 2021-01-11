# Optimizing an ML Pipeline in Azure

## Overview

This project is part of the Udacity Azure ML Nanodegree.
In this project, we build and optimize an Azure ML pipeline using the Python SDK and a provided Scikit-learn model.
This model is then compared to an Azure AutoML run.

#### Figure 1: Workflow of the project : ####

![](images/project_workflow.png)

## Summary

The Bank Marketing dataset used for this project contains data about bank customers using which we seek to predict whether a customer will subscribe to the fixed term deposit or not.
The best performing model was a Voting Ensemble model produced by Auto ML run and it had an accuracy of 0.91648.
 
## Scikit-learn Pipeline

Firstly the data was taken from the UCI Bank Marketing dataset. Then the datset was passed to `clean_data` function in `train.py` file to clean the data and apply one-hot encoding. Then the dataset was split into train and test set. After that the training data was passed to a **Logistic Regression Model**.

The hyperparameters of the logistic regression model such as Inverse of Regularization strength (`C`) and Maximum Number of Iterations (`max_iter`) were tuned using Microsoft Azure Machine Learning's hyperparameter tuning package **HyperDrive**. 

Then `RandomParameterSampling` was used as a parameter sampler. In this sampling algorithm, hyperparameter values are randomly selected from the defined search space and supports early termination of low-performance runs thus taking less computational efforts.

Here `BanditPolicy` was used as a stopping policy. Bandit policy is based on slack factor/slack amount and evaluation interval. Bandit terminates runs where the primary metric is not within the specified slack factor/slack amount compared to the best performing run thus saving computational time.

#### Figure 2 : Azure ML Studio Hyper Drive Run Details - ####

![](images/HyperDriveRunDetails.png)

#### Figure 3 : Azure ML Studio Hyper Drive Run Metrics - 

![](images/HyperDriveRunMetrics.png)

#### Figure 4 : Azure ML Studio Hyper Drive Running Details from notebook - ####

![](images/HyperDriveRunningDetails.png)

## AutoML

AutoML is used to automate the repetitive tasks by creating a number of pipelines in parallel that try different algorithms and parameters. This iterates through ML algorithms paired with feature selections, where each iteration produces a model with a training score. The higher the score, the better the model is considered to fit the data. This process terminates when the exit criteria defined in the experiment is satisfied.

The Banking Datset was pre-processed using `clean_data` and the trained on the `AutoMLConfig` with parameters :
`experiment_timeout_minutes`: 30
`task`: classification
`primary_metric`: accuracy
`training_data`: Tabular dataset created from csv data file using TabularDatasetFactory
`compute_target`: compute cluster
`label_column`: y(result)
`n_cross_validations`: 5

In this the **Voting Ensemble** Model performed the best with an accuracy of `0.91648`.

## Pipeline comparison

The accuracy received from the HyperDrive Model was ___ whereas the accuracy received from the AutoML model was ___ . Thus the AutoML model outperformed our HyperDrive model by difference of __ accuracy. 

This difference of accuracy was mainly because of the architecture of the two algorithms. In the HyperDrive case we fixed the model to be Logistic Regression and went on to select the best hyperparameters by using HyperDrive. Whereas AutoML gave us a lot of flexibility by selecting the model itself from a range of models and tuning its hyperparameters. 

Thus we can use HyperDrive when we know the model and have less time and computation power, and AutoMl can be used to get more accurate model by giving more time and computational power.

## Future work

Some areas of improvement for future experiments are:

* Class balancing - The dataset used for training the model was imbalanced with size of smallest class = 3692 out of 32950 training samples. This imbalanced data can lead to a falsely perceived positive effect of a model's accuracy because the input data has bias towards one class. Thus the training data can be made more balanced to remove this bias.

* Using Deep Learning to get better Accuracy.

* Using different combination of the C and max-iter values used in the HyperDrive run.

* Increasing the `experiment_timeout_minutes` inthe AutoML run.

* Using different values of `n_cross_validations` in AutoML run.

## Proof of cluster clean up

The compute cluster created can be deleted by running the following code:
`cpu_cluster.delete()`

OR Directly from the Delete option - 

![](images/DeletingComputeCluster.png)
