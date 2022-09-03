# Predict Customer Churn

- Project **Predict Customer Churn** of ML DevOps Engineer Nanodegree Udacity


## Project Description

This project aims to predict the probability a customer will churn. 


## Files and data description
- `main.py`: main python script to execute the whole pipeline: import data -> EDA -> feature engineering -> train model
- `churn_library.py`: module containing core functions to be used in the `main.py` file
- `churn_script_logging_and_tests.py`: module to test all functions in `main.py`
- `requirements.txt`: text file containing all the packages to be installed in order to execute the `main.py` script smoothly 
- ./data: data to train and test model
- ./images: plots and reports from EDA and model training process
- ./models: trained models objects in the form of pickle files


## Running Files
- Create a virtual environment using conda or pyenv
- clone the repository
- run `pip3 install -r requirements.txt` to install the all the dependencies
- run `pytest .` to make sure all the functions work as expected
- run `python3 main.py` to execute the whole pipeline. This results in EDA figures as well as classification reports/plots to be saved to the `./images` folder as well as model (in the form of pickle files) to be saved in `./models` folder



