Kaggle Competition: Santander Product Recommendation

Authors:
Rakan Frijat 
Ian Commerford

------------------------------------------------------------------------------------------------------------------------------
pre-processing.py

The purpose of this file is to take the raw Kaggle data and read it into a pandas dataframe.  
It performs some preprocessing such as data-cleansing, reformatting and creation of lag variables.


Instructions:


    1. Download the zip file from the kaggle website, and place it in the same directory as the script.

https://www.kaggle.com/c/santander-product-recommendation/data


    2. Run the script   (note: you can comment out the function calls at the bottom if necessary)
    
python pre-processing.py santander-product-recommendation/ train_ver2.csv test_ver2.csv santander-product-recommendation.zip

    Details of arguments:
    - Filepath: the directory to read/write data from.   Must have a '/' afterwards.
    - Training File: the name of the training file contained in the kaggle zip
    - Test File: the name of the test file contained in the kaggle zip
    - Downloaded Zip file: the name of the downloaded kaggle zip



------------------------------------------------------------------------------------------------------------------------------
models.py

This file contains all the models that can be executed in a single file.  To run different models, before executing adjust
the setting variables as detailed below.


Instructions:


    1. Run Pre-processing.py

    
    2. Adjust setting variables by modifying models.py depending on choice of model
    
Settings for Decision Tree 
DISCRETISE_CONTINUOUS = True
CLASSIFIER = 'DT'

Settings for Random Forest 
DISCRETISE_CONTINUOUS = True
CLASSIFIER = 'RF'

Settings for XGBoost
DISCRETISE_CONTINUOUS = True
CLASSIFIER = 'XGB'

Settings for Multilabel Bernoulli Naive Bayes
DISCRETISE_CONTINUOUS = True
CLASSIFIER = 'BNB'

Settings for Logistic Regression  
DISCRETISE_CONTINUOUS = False
CLASSIFIER = 'LR'


    3. Run the script
    
python pre-processing.py santander-product-recommendation/

    Details of arguments:
    - Filepath: the directory to read/write data from.   Must have a '/' afterwards.