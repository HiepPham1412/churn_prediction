from cmath import log
import os
import logging
import churn_library as cls
import pytest
import pandas as pd
import numpy as np
from pandas.testing import assert_frame_equal

logging.basicConfig(
	filename='./logs/churn_library.log',
	level = logging.INFO,
	filemode='w',
	format='%(name)s - %(levelname)s - %(message)s')

@pytest.fixture()
def input_df():
	df = pd.read_csv("./data/bank_data.csv")
	df = df.head(1000)
	return df


def test_import():
	'''
	test data import - this example is completed for you to assist with the other test functions
	'''
	try:
		df = cls.import_data("./data/bank_data.csv")
		logging.info("Testing import_data: SUCCESS")
	except FileNotFoundError as err:
		logging.error("Testing import_eda: The file wasn't found")
		raise err

	try:
		assert df.shape[0] > 0
		assert df.shape[1] > 0
	except AssertionError as err:
		logging.error("Testing import_data: The file doesn't appear to have rows and columns")
		raise err


def test_eda(input_df):
	'''
	test perform eda function
	'''
	df = cls.perform_eda(input_df)

	try:
		assert len(df) == len(input_df)
		logging.info('Sucess: EDA process does not aritficially inflate the data')
	except AssertionError:
		logging.error('EDA process inflated the data')

	try:
		assert os.path.isfile("./images/churn_distribution.png") is True
		logging.info('Churn distribution file found')
	except AssertionError:
		logging.error('No churn distribution file found')

	try:
		assert os.path.isfile("./images/correlation.png") is True
		logging.info('Correlation heatmap file found')
	except AssertionError:
		logging.error('No correlation heatmap file found')

	try:
		assert os.path.isfile("./images/martial_status.png") is True
		logging.info('martial_status file found')
	except AssertionError:
		logging.error('No correlation heatmap file found')

	try:
		assert os.path.isfile("./images/total_trans_cost.png") is True
		logging.info('total_trans_cost file found')
	
	except AssertionError:
		logging.error('No total_trans_cost file found')


def test_encoder_helper(input_df):
	'''
	test encoder helper
	'''
	input_df = cls.perform_eda(input_df)
	category_lst = ["Gender", "Education_Level", "Marital_Status", "Income_Category", "Card_Category"]
	try:
		assert set(category_lst).issubset(input_df.columns.tolist())
		logging.info('Input data contains columns in the category_lst')

	except AssertionError:
		logging.error('input data frame does not contain necessary columns: "Gender", "Education_Level", "Marital_Status", "Income_Category", "Card_Category"')

	df = cls.encoder_helper(input_df, category_lst)

	try:
		assert len(df) == len(input_df)
		assert len(df.columns.tolist()) == len(input_df.columns.tolist()) + len(category_lst)
		assert set([col + '_Churn' for col in category_lst]).issubset(df.columns.tolist())
		logging.info('Sucess: expected shape and columns of the output data')

	except AssertionError:
		logging.error('No or unexpected new columns/rows created')
	
	try:
		assert_frame_equal(input_df, cls.encoder_helper(input_df, []))
		logging.info('Sucess: category_lst=[]')

	except AssertionError:
		logging.error('Error with empty category_lst')




def test_perform_feature_engineering(input_df):
	'''
	test perform_feature_engineering
	'''

	try:
		assert set(['Churn']).issubset(input_df.columns.tolist())
		logging.info('Response variable Churn is in the data')
	
	except AssertionError:
		logging.error('Response variable Churn is not the data')


	X_train, X_test, y_train, y_test = cls.perform_feature_engineering(df=input_df, response='Churn')

	# expected input, output shape
	try:
		assert len(X_train) + len(X_test) == len(input_df)
		assert len(y_train) + len(y_test) == len(input_df)
		assert len(X_train) == len(y_train)
		assert len(X_test) == len(y_test)
		assert X_train.shape[1] == X_test.shape[1]
		logging.info('Sucess: expected shape of the output data')

	except AssertionError:
		logging.error('Uxpected shape of the output data')

	

def test_train_models(input_df):
	'''
	test train_models
	'''
	X_train, X_test, y_train, y_test = cls.perform_feature_engineering(df=input_df, response='Churn')
	cls.train_models(X_train, X_test, y_train, y_test)

	try:
		assert os.path.isfile("./images/classification_report_logistic_regression.png")
		assert os.path.isfile("./images/classification_report_random_forest.png")
		logging.info('Sucess: classification reports generated')
	except AssertionError:
		logging.info('Classification reports are not generated')

	try:
		assert os.path.isfile("./images/feature_importance.png")
		logging.info('Sucess: feature_importance chart generated')
	except AssertionError:
		logging.info('feature_importance chart is not generated')

	try:
		assert os.path.isfile("./images/roc_auc_logistic_regression_test.png")
		assert os.path.isfile("./images/roc_auc_random_forest_test.png")
		logging.info('Sucess: ROC AUC charts are generated')
	except AssertionError:
		logging.info(' ROC AUC charts are not generated')

	try:
		assert os.path.isfile("./models/logistic_model.pkl")
		assert os.path.isfile("./images/rrfc_model.pkl")
		logging.info('Sucess: model pickel files are created and saved')
	except AssertionError:
		logging.info('model pickel files are not created or saved')

if __name__ == "__main__":
	test_import(input_df)
	test_eda(input_df)
	test_encoder_helper(input_df)
	test_perform_feature_engineering(input_df)
	test_train_models(input_df)