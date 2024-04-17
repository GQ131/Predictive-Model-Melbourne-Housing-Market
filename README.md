# Melbourne-Housing-Market

This project explores and analyzes the Melbourne housing market through a comprehensive dataset, aiming to predict house prices using machine learning models. The dataset, Melbourne_housing.csv, contains various features related to properties sold in Melbourne, including suburb, address, number of rooms, price, and many others.
Project Structure:

## Data Exploration and Visualization:
Initial examination of the dataset to understand the distribution of different features. Utilized visualizations like histograms, scatter plots, and box plots to uncover insights into the housing market's dynamics.

## Linear Regression Model Development:
Created a Linear Regression model to predict the prices of houses. This involved preprocessing steps such as handling missing values, encoding categorical variables, and feature selection, followed by an explanation of the model development process.

## Model Evaluation:
Assessed the model's performance using various metrics, split the dataset into training and testing sets for a robust evaluation, and provided interpretations of these metrics

## Regularization with Lasso:
Explored the need for regularization to address overfitting. Applied Lasso regularization, tuning the alpha parameter for optimal performance, and compared the results with the base Linear Regression model.

## Out-of-Sample Performance Analysis:
Ignoring the previous models, split the data anew and evaluated both Linear and Lasso Regression models' performance through AIC, AICc, BIC metrics, and 5-fold cross-validation. Analyzed the true out-of-sample (OOS) performance by comparing the deviance on test data.

## Key Insights:

    - Discovered how various features impact house prices in Melbourne.
    - Evaluated the effectiveness of Linear and Lasso Regression models in predicting house prices.
    - Assessed model performance using in-sample and out-of-sample evaluation metrics.

## Technologies Used:

    - Python
    - Pandas for data manipulation
    - Scikit-learn for model building and evaluation
    - Matplotlib and Seaborn for data visualization

This repository contains all the code, visualizations, and analysis performed as part of the assignment. The code is well-commented and formatted for easy understanding and reproducibility.
