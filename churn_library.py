"""
Module to run import, perform eda, preprocess the data and train classifiers to predict Churn
Author: Hiep Pham
Date: 2022-09-01
"""

# library doc string
import joblib
import numpy as np
import pandas as pd
from pandas.plotting import table
import logging
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import plot_roc_curve, classification_report

# import libraries
import os

os.environ["QT_QPA_PLATFORM"] = "offscreen"


def import_data(pth):
    """
    returns dataframe for the csv found at pth

    input:
                    pth: a path to the csv
    output:
                    df: pandas dataframe
    """
    try:
        df = pd.read_csv(pth)
        df = df.head(200)
        return df

    except AttributeError:

        logging.info("File does not exits")


def perform_eda(df):
    """
    perform eda on df and save figures to images folder
    input:
                    df: pandas dataframe

    output:
                    None
    """
    # churn rate
    fig, ax = plt.subplots(1, 1, figsize=(20, 10))
    df["Churn"] = df["Attrition_Flag"].apply(
        lambda val: 0 if val == "Existing Customer" else 1
    )
    ax.hist(df["Churn"])
    plt.savefig("./images/churn_distribution.png", format="png")

    # Marital_Status
    df.Marital_Status.value_counts("normalize").plot(kind="bar")
    plt.savefig("./images/martial_status.png", format="png")

    # Total_Trans_Ct
    sns.histplot(df["Total_Trans_Ct"], stat="density", kde=True)
    plt.savefig("./images/total_trans_cost.png", format="png")

    # correlation
    sns.heatmap(df.corr(), annot=False, cmap="Dark2_r", linewidths=2)
    plt.savefig("./images/correlations.png", format="png")
    plt.close()

    return df


def perform_feature_engineering(df, response):
    """
    input:
                    df: pandas dataframe
                    response: string of response name [optional argument that could be used for naming variables or index y column]

    output:
                    X_train: X training data
                    X_test: X testing data
                    y_train: y training data
                    y_test: y testing data
    """
    df["Churn"] = df["Attrition_Flag"].apply(
        lambda val: 0 if val == "Existing Customer" else 1
    )
    cat_columns = [
        "Gender",
        "Education_Level",
        "Marital_Status",
        "Income_Category",
        "Card_Category",
    ]
    df = encoder_helper(df, category_lst=cat_columns)

    feat_cols = [
        "Customer_Age",
        "Dependent_count",
        "Months_on_book",
        "Total_Relationship_Count",
        "Months_Inactive_12_mon",
        "Contacts_Count_12_mon",
        "Credit_Limit",
        "Total_Revolving_Bal",
        "Avg_Open_To_Buy",
        "Total_Amt_Chng_Q4_Q1",
        "Total_Trans_Amt",
        "Total_Trans_Ct",
        "Total_Ct_Chng_Q4_Q1",
        "Avg_Utilization_Ratio",
        "Gender_Churn",
        "Education_Level_Churn",
        "Marital_Status_Churn",
        "Income_Category_Churn",
        "Card_Category_Churn",
    ]
    X = df[feat_cols]
    y = df[response]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    return X_train, X_test, y_train, y_test


def encoder_helper(df, category_lst):
    """
    helper function to turn each categorical column into a new column with
    propotion of churn for each category - associated with cell 15 from the notebook

    input:
                    df: pandas dataframe
                    category_lst: list of columns that contain categorical features

    output:
                    df: pandas dataframe with new columns for
    """
    for category in category_lst:
        # gender encoded column
        val_lst = []
        mean_groups = df.groupby(category).mean()["Churn"]
        for val in df[category]:

            val_lst.append(mean_groups.loc[val])

        df[category + "_Churn"] = val_lst

    return df


def classification_report_image(
        y_train,
        y_test,
        y_train_preds_lr,
        y_train_preds_rf,
        y_test_preds_lr,
        y_test_preds_rf,
):
    """
    produces classification report for training and testing results and stores report as image
    in images folder
    input:
                    y_train: training response values
                    y_test:  test response values
                    y_train_preds_lr: training predictions from logistic regression
                    y_train_preds_rf: training predictions from random forest
                    y_test_preds_lr: test predictions from logistic regression
                    y_test_preds_rf: test predictions from random forest

    output:
                    None
    """
    # Logistic regression
    plt.rc('figure', figsize=(7, 5))
    plt.text(0.01, 1.25, str('Logistic Regression Train'),
             {'fontsize': 10}, fontproperties='monospace')
    plt.text(0.01, 0.05, str(classification_report(y_train, y_train_preds_lr)), {
             'fontsize': 10}, fontproperties='monospace')  # approach improved by OP -> monospace!
    plt.text(0.01, 0.6, str('Logistic Regression Test'), {
             'fontsize': 10}, fontproperties='monospace')
    plt.text(0.01, 0.7, str(classification_report(y_test, y_test_preds_lr)), {
             'fontsize': 10}, fontproperties='monospace')  # approach improved by OP -> monospace!
    plt.axis('off')
    plt.savefig(
        "./images/classification_report_logistic_regression.png",
        format="png")
    plt.close('all')

    # Random Forest
    plt.rc('figure', figsize=(7, 5))
    plt.text(0.01, 1.25, str('Random Forest Train'), {
             'fontsize': 10}, fontproperties='monospace')
    plt.text(0.01, 0.05, str(classification_report(y_test, y_test_preds_rf)), {
             'fontsize': 10}, fontproperties='monospace')  # approach improved by OP -> monospace!
    plt.text(0.01, 0.6, str('Random Forest Test'), {
             'fontsize': 10}, fontproperties='monospace')
    plt.text(0.01, 0.7, str(classification_report(y_train, y_train_preds_rf)), {
             'fontsize': 10}, fontproperties='monospace')  # approach improved by OP -> monospace!
    plt.axis('off')
    plt.savefig(
        "./images/classification_report_random_forest.png",
        format="png")

    plt.close('all')

    return None


def feature_importance_plot(model, X, output_pth):
    """
    creates and stores the feature importances in pth
    input:
                    model: model object containing feature_importances_
                    X_data: pandas dataframe of X values
                    output_pth: path to store the figure

    output:
                    None
    """
    # Calculate feature importances
    importances = model.best_estimator_.feature_importances_
    # Sort feature importances in descending order
    indices = np.argsort(importances)[::-1]

    # Rearrange feature names so they match the sorted feature importances
    names = [X.columns[i] for i in indices]

    # Create plot
    plt.figure(figsize=(20, 5))

    # Create plot title
    plt.title("Feature Importance")
    plt.ylabel('Importance')

    # Add bars
    plt.bar(range(X.shape[1]), importances[indices])

    # Add feature names as x-axis labels
    plt.xticks(range(X.shape[1]), names, rotation=90)
    plt.savefig(output_pth)

    return None


def roc_auc_plot(lr_model, rf_model, X_train, y_train, X_test, y_test):
    """
    produces ROC AUC ploet for training and testing results and stores image in images folder
    input:	lr_model: logistic regression model object
                    rf_model:  random forest model object
                    X_train: training features
                    X_test:  test features
                    y_train: training response values
                    y_test:  test response values

    output:
                    None
    """
    # Logistic Regression
    plot_roc_curve(lr_model, X_train, y_train)
    plt.savefig("./images/roc_auc_logistic_regression_train.png", format="png")

    plot_roc_curve(lr_model, X_test, y_test)
    plt.savefig("./images/roc_auc_logistic_regression_test.png", format="png")

    # Random Forest
    plot_roc_curve(rf_model, X_train, y_train)
    plt.savefig("./images/roc_auc_random_forest_train.png", format="png")

    plot_roc_curve(rf_model, X_test, y_test)
    plt.savefig("./images/roc_auc_random_forest_test.png", format="png")

    return None


def train_models(X_train, X_test, y_train, y_test):
    """
    train, store model results: images + scores, and store models
    input:
                    X_train: X training data
                    X_test: X testing data
                    y_train: y training data
                    y_test: y testing data
    output:
                    None
    """
    # grid search

    rfc = RandomForestClassifier(random_state=42)
    # Use a different solver if the default 'lbfgs' fails to converge
    # Reference:
    # https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
    lrc = LogisticRegression(solver="lbfgs", max_iter=3000)

    param_grid = {
        "n_estimators": [200, 500],
        "max_features": ["auto", "sqrt"],
        "max_depth": [4, 5, 100],
        "criterion": ["gini", "entropy"],
    }

    cv_rfc = GridSearchCV(estimator=rfc, param_grid=param_grid, cv=5)
    cv_rfc.fit(X_train, y_train)

    lrc.fit(X_train, y_train)

    y_train_preds_rf = cv_rfc.best_estimator_.predict(X_train)
    y_test_preds_rf = cv_rfc.best_estimator_.predict(X_test)

    y_train_preds_lr = lrc.predict(X_train)
    y_test_preds_lr = lrc.predict(X_test)

    classification_report_image(
        y_train,
        y_test,
        y_train_preds_lr,
        y_train_preds_rf,
        y_test_preds_lr,
        y_test_preds_rf,
    )
    feature_importance_plot(cv_rfc, X_train, './images/feature_importance.png')
    roc_auc_plot(lrc, cv_rfc.best_estimator_, X_train, y_train, X_test, y_test)

    joblib.dump(cv_rfc.best_estimator_, './models/rfc_model.pkl')
    joblib.dump(lrc, './models/logistic_model.pkl')
