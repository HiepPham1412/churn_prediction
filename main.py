from churn_library import import_data, perform_eda, perform_feature_engineering, train_models


# step 1: import data
df = import_data(pth='data/bank_data.csv')

# step 2: perform EDA and save figures in ./images/
perform_eda(df)

# step 3: perform feature engineering
X_train, X_test, y_train, y_test = perform_feature_engineering(df=df, response='Churn')

# step 4: train model, save classification reports and model object to ./models/
train_models(X_train, X_test, y_train, y_test)
