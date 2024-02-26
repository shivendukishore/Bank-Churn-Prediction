# Churn Modelling - How to predict if a bank’s customer will stay or leave the bank

Using a source of 165034 bank records,  we applied machine learning models to predict the likelihood of customer churn. We accomplished this using the following steps:

## 1. Clean the data

By reading the dataset into a dataframe using pandas,  we removed unnecessary data fields including individual customer IDs and names. This left us with a list of columns for Credit Score, Geography, Gender, Age, Length of time as a Bank customer, Balance, Number Of Bank Products Used, Has a Credit Card, Is an Active Member, Estimated Salary and Exited. 

## 2. Analyze initial DataFrame

Utilizing Matplotlib, Seaborn and Pandas, we next analyzed the data. We can see that our dataset was imbalanced. The majority class, "Stays" (0), has around 80% data points and the minority class, "Exits" (1), has around 20% datapoints. To address this, we StratifiedKfold in our machine learning algorithms. More on those later on. 

In percentage, female customers are more likely to leave the bank at 25%, compared to 16% of males.

The smallest number of customers are from Germany, and they are also the most likely to leave the bank. Almost one in three German customers in our sample left the bank.

## 3. Machine Learning using different models

We tested different machine learning models (and used six in the final application) to predict customer churn, including Logistic Regression, Random Forest and XGBoost. 

As mentioned earlier, we also used StratifiedKfold in cross validating models to handle issues with the imbalanced data, incorporating Stratified K-Fold cross-validation in our project's machine learning algorithms ensures a balanced representation of the target variable across folds, mitigating the impact of class imbalance. This enhancement promotes robust model training and evaluation, fostering reliable performance metrics.


## Dataset

- [Kaggle - Churn Modelling Classification Data Set](https://www.kaggle.com/shrutimechlearn/churn-modelling)

For a deeper dive into our analysis and process, take a look at our [full presentation](https://github.com/shivendukishore/Bank-Churn-Prediction/blob/main/Bank%20Churn%20Predictions%20final.pdf).

 
