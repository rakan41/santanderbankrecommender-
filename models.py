"""
Kaggle Competition: Santander Product Recommendation
Authors:

Rakan Frijat
Ian Commerford

Python version: 3.6

Program Arguments:
Argument 1: Directory from where to read data from
Sample argument: data/

Program Run Instructions:
The model and data preparation options are available in "Program Run Parameters" section. Change the value for the
parameters in capitals to change the way the model or data preparation is implemented. The avaialble options are
listed as comments.

"""

# Import Modules
import sys
import pandas as pd
import numpy as np
import os
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn import preprocessing
from sklearn.naive_bayes import BernoulliNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
from sklearn.neighbors.kde import KernelDensity
import ml_metrics as metrics
from sklearn.multiclass import OneVsRestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import zipfile

# Program Arugments
args = sys.argv
filepath = args[1]      # the directory to read and write data to/from


# Program Run Parameters

CLASSIFIER = 'BNB'              # The classifier model to run. Available options are BNB, LR, DT, XGB, RF, KNN
DISCRETISE_CONTINUOUS = True    # Select true if you wish to convert continuous variables into discrete
USE_DUMMIES = True              # Convert categorical variables into dummy variables. Leave this as true for most models
ALL_PARTITIONS = False          # Select true if you wish to train over all available partitions.
ALL_LABELS = True               # Select false if you wish to only classifiy the top 14 products
PURCHASED_ONLY = False          # Select true if you wish to train on only customers who have purchased something
CLUSTER_METHOD = 'kmeans'       # Method of discretising continuous variables. Options are kmeans, kde, cut.
SCALE_VARIABLES = False         # Select true to scale continuous variables (either Gaussian or minmax)
CLIP_OUTLIERS = True            # Select true to put an upper bound of values 3 standard deviations from the mean.
ADJUST_MULTIPLE_PURCHASES = False   # melt output labels into one dimension

# DEFINING FUNCTIONS

# Helper function that one-hot encodes categorical varibles within dataframes
# Takes columns across dataframes to ensure they are consistently encoded
def get_onehot(df1, df2, col):
    # find list of all unique values
    unique_list = df1[col].unique().to_list() + df2[col].unique().to_list()
    # fit label
    le = preprocessing.LabelEncoder()
    le.fit(unique_list)    # update df columns with encoded values
    df1[col] = le.transform(df1[col])
    df2[col] = le.transform(df2[col])


# Function that removes outliers
def clip_outliers(df1, df2, col):
    z = np.abs(stats.zscore((df1[col])))
    z2 = np.abs(stats.zscore((df2[col])))
    df1.loc[(z > 3), col] = df1[col].quantile(0.99)
    df2.loc[(z2 > 3), col] = df2[col].quantile(0.99)

# Helper function that takes a continuous column and converts to an ordinal encoding
def discretise(df1, df2, col, method='kmeans', scale=None):
    print("Discretising {} using {} method".format(col, method))

    # scale variables
    if scale=='norm':
        df1[col] = (df1[col] - df1[col].mean()) / df1[col].std()
        df2[col] = (df2[col] - df2[col].mean()) / df2[col].std()
    elif scale=='minmax':
        df1[col] = (df1[col] - df1[col].min()) / (df1[col].max() - df1[col].min())
        df2[col] = (df2[col] - df2[col].min()) / (df2[col].max() - df2[col].min())

    # discretising method
    if method=='cut':
        df1[col] = pd.cut(df1[col], bins=4, labels=['1', '2', '3', '4'])
        df2[col] = pd.cut(df2[col], bins=4, labels=['1', '2', '3', '4'])
    elif method=='kmeans':
        # create K means model
        km = KMeans(n_clusters=4)
        # reshape dfs to arrays
        values1 = np.array(df[col]).reshape(-1, 1)
        values2 = np.array(df2[col]).reshape(-1, 1)
        # fit kmeans models
        km.fit(np.concatenate((values1, values2), axis=0))
        # encode variables
        df1[col] = km.predict(values1).reshape(-1,1)
        df2[col] = km.predict(values2).reshape(-1,1)
    elif method=='kde':
        # create K means model
        km = KernelDensity(n_clusters=4)
        # reshape dfs to arrays
        values1 = np.array(df[col]).reshape(-1, 1)
        values2 = np.array(df2[col]).reshape(-1, 1)
        # fit kernal density
        km.fit(np.concatenate((values1, values2), axis=0))
        # encode variables
        df1[col] = km.predict(values1).reshape(-1,1)
        df2[col] = km.predict(values2).reshape(-1,1)

# Helper function that takes categorical variables and converts them to dummies
def get_dummies(df1, df2, cat_features):
    # generate dataframes with dummy variables
    df1_dummies = (pd.get_dummies(df1[cat_features], drop_first=False))
    df2_dummies = (pd.get_dummies(df2[cat_features], drop_first=False))
    # append new columns to existing dataframes
    df1 = pd.concat([df1, df1_dummies], axis=1)
    df2 = pd.concat([df2, df2_dummies], axis=1)
    # store new list of dummy features
    dummy_features = df1_dummies.columns.to_list()
    return df1, df2, dummy_features

# Function that takes model predictions and outputs a submission file
def output_submission(df_eval, eval_probs, recs, labels):
    print("Creating submission file.")
    labels = [col.replace("_new", "") for col in labels]
    # organising output into a dataframe
    output = pd.DataFrame()
    output['ncodpers'] = df_eval.index.get_level_values('customer_code').to_list()
    output['ncodpers'] = output['ncodpers'].astype(int)
    temp = [[labels[r] for r in rec] for rec in recs]
    output['added_products'] = np.array([" ".join(x) for x in temp])
    # write submission file to csv and zip it
    output.to_csv("{}submission.csv".format(filepath), index=False)
    with zipfile.ZipFile('{}test_output.zip'.format(filepath), 'w') as myzip:
        myzip.write("{}submission.csv".format(filepath), compress_type=zipfile.ZIP_DEFLATED)
    print("Submission file completed.")
    return output

# Calculates Mean Average Precision across customers
# Parameters:
# y_true: numpy array of the true labels (i.e. products purchased for that period)
# y_scores: numpy array of label prediction probabilities, used to create a recommendation list
def get_MAP(y_true, y_scores, k=7):
    print("Getting MAP scores.")
    # generating a numpy array product recommendations
    recs = (-y_scores).argsort()[:, :7]
    # initalising scores
    score = 0
    count = 0

    # calculate mean average precision for each user
    for i in range(y_true.shape[0]):
        count += 1
        # find column indexes for any products actually bought
        if np.argwhere(y_true[i] == True).shape[0] > 0:
            choices = np.argwhere(y_true[i] == True)[0]
        else:
            continue
        # selecting users recommendations
        top_k = recs[i]
        if choices.shape[0] == 0 or top_k.shape[0] == 0:
            continue
        else:
            # calculate metric to compare the actual choices with recommendations
            l_top_items = top_k.tolist()
            l_choices = choices.tolist()
            current = 0
            num_items_found = 0
            numerator = 0
            denominator = 0
            for guess in l_top_items:
                denominator += 1
                if guess in l_choices:
                    num_items_found += 1
                    numerator += 1
                    current += numerator/denominator
            # increase score if at least 1 of the recommendations matched the actual choice
            if num_items_found > 0:
                score += current / num_items_found
    # return map score
    return score / count

# MODELS
#  Logistic Regression
def get_LR(X_train, X_test, Y_train):
    print("Fitting Logistic Regression model.")
    model = OneVsRestClassifier(LogisticRegression(solver='lbfgs'))
    model.fit(X_train, Y_train)
    print("Completed.")
    return model

# Multilabel Bernoulli Naive Bayes Classifier
def get_BNB(X_train, X_test, Y_train):
    print("Fitting BNB model.")
    # fit Bernoulli classifier with one vs rest multilabel strategy
    model = OneVsRestClassifier(BernoulliNB())
    model.fit(X_train, Y_train)
    print("Complete.")
    return model

# Random Forest Classifier
def get_RF(X_train, X_test, Y_train):
    print("Fitting Random Forest model.")
    model = OneVsRestClassifier(RandomForestClassifier(n_estimators=100, random_state=0, min_samples_leaf=500))
    model.fit(X_train, Y_train)
    print("Completed.")
    return model

# KNN Classifier
def get_KNN(X_train, X_test, Y_train, n=5):
    print("Fitting KNN model with {} neighbours.".format(n))
    model = OneVsRestClassifier(KNeighborsClassifier(n_neighbors=n))
    model.fit(X_train, Y_train)
    print("Completed.")
    return model

# XGBoost Classifier
def get_XGB(X_train, X_test, Y_train):
    print("Fitting XG Boost Model")
    classes = Y_train.shape[1]
    model = OneVsRestClassifier(xgb.XGBClassifier(max_depth=6, max_delta_step=5,
                                                  booster='gbtree', learning_rate=0.05, n_estimators=15))
    model.fit(X_train, Y_train)
    print("Completed.")
    return model

# Decision Tree Classifier
def get_DT(X_train, X_test, Y_train):
    print("Fitting Decision Tree model.")
    model = OneVsRestClassifier(DecisionTreeClassifier(max_depth=4, min_samples_leaf=250))
    model.fit(X, Y)
    print("Completed.")
    return model

# REFERENCE
# list of available products and their lagged versions
products = ['ind_ahor_fin_ult1', 'ind_aval_fin_ult1', 'ind_cco_fin_ult1', 'ind_cder_fin_ult1',
          'ind_cno_fin_ult1', 'ind_ctju_fin_ult1', 'ind_ctma_fin_ult1', 'ind_ctop_fin_ult1',
          'ind_ctpp_fin_ult1', 'ind_deco_fin_ult1', 'ind_deme_fin_ult1', 'ind_dela_fin_ult1', 'ind_ecue_fin_ult1',
          'ind_fond_fin_ult1', 'ind_hip_fin_ult1', 'ind_plan_fin_ult1', 'ind_pres_fin_ult1', 'ind_reca_fin_ult1',
          'ind_tjcr_fin_ult1', 'ind_valo_fin_ult1', 'ind_viv_fin_ult1', 'ind_nomina_ult1', 'ind_nom_pens_ult1', 'ind_recibo_ult1']

products_t_1 = [product + "_t_1" for product in products]
products_t_2 = [product + "_t_2" for product in products]
products_t_3 = [product + "_t_3" for product in products]

# dependent variable
all_labels = [product + "_new" for product in products]

# categorical variables to encode
cat_features = ['cust_segment', 'sex', 'cust_type', "cust_relation_type"]

# continuous variables to discretise
cont_features = ['age', 'cust_income', 'customer_duration', 'total_products_t_1', 'total_added_t_1', 'total_added_t_2', 'total_added_t_3']

# DATA PREPARATION
# read training data
if ALL_PARTITIONS == True:
    files = os.listdir('{}transformedfiles/'.format(filepath))
    files.sort()
    print("Merging training partitions.")
    df = pd.read_pickle("{}transformedfiles/{}".format(filepath, files[0]), compression='zip')
    for i in range(9, len(files)-1-1):
        df2 = pd.read_pickle("{}transformedfiles/{}".format(filepath, files[i + 1]), compression='zip')
        df = pd.concat([df, df2], axis=0)
else:
    df = pd.read_pickle("{}transformedfiles/2016-05-28.pickle.zip".format(filepath))
# read evaluation data
df_eval = pd.read_pickle("{}transformedfiles/2016-06-28.pickle.zip".format(filepath))

if ADJUST_MULTIPLE_PURCHASES == True:
    index_cols = ['customer_code', 'part_dt']
    # create a new column with number of new products, and get subset of df to change where total_new >=2
    df['total_new'] = df[all_labels].sum(axis=1)
    df_unchanged=df[df['total_new']<=1].reset_index()
    df_changed = df[df['total_new']>=2].reset_index()
    # melt all the product columns into a single column, thereby duplicating rows for each new product
    df2 = df_changed[index_cols + all_labels].melt(id_vars=index_cols, var_name='New_Product',
                                                                 value_name='Added_bool')
    # filter out False values
    df2=df2[df2['Added_bool']]
    # create temporary key and then pivot back into multiple columns
    df2['temp'] = df2['customer_code'].astype(str) + df2['part_dt'].astype(str) + df2['New_Product']
    df3 = df2.pivot(index='temp', columns='New_Product', values='Added_bool')
    df4 = df2.merge(df3, on='temp')
    df4 = df4.drop(columns=['temp','Added_bool','New_Product'])
    # merge pivot back into df_changed
    df_changed = df_changed.drop(columns=all_labels)
    df_changed = df4.merge(df_changed,on='customer_code')
    # merge all back into one df
    df = df_unchanged.append(df_changed, ignore_index=True, sort=False)
    df = df.drop(columns='total_new')
    df[all_labels] = df[all_labels].fillna(False)


# dropping labels that are rarely used
if ALL_LABELS == True:
    labels = all_labels
else:
    x = df[all_labels].sum().to_dict()
    labels = sorted(x, key=x.get, reverse=True)[:15]


# encode categorical variables
for col in cat_features:
    get_onehot(df, df_eval, col)

# clip outliers
if CLIP_OUTLIERS == True:
    for col in cont_features:
        clip_outliers(df, df_eval, col)

# descretise continuous variables
if DISCRETISE_CONTINUOUS == True:
    for col in cont_features:
        discretise(df, df_eval, col, method=CLUSTER_METHOD, scale=SCALE_VARIABLES)

# convert categorical variables
if USE_DUMMIES == True:
    if DISCRETISE_CONTINUOUS == True:
        cat_features = cat_features + cont_features
        cont_features = []
    df, df_eval, cat_features = get_dummies(df, df_eval, cat_features)

# select features to train model
features = products_t_1 + cat_features + cont_features

# only use customers that have bought something for training
if PURCHASED_ONLY == True:
    df_cut = df.loc[df['total_added_t_1'] != 0]

# remove duplicate columns
df = df.loc[:,~df.columns.duplicated()]
df_eval = df_eval.loc[:,~df_eval.columns.duplicated()]

# set up training and evaluation sets
X = df[features]
X_eval = df_eval[features]
Y = df[labels]
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)


# MODELS
# train model
if CLASSIFIER == 'LR':
    model = get_LR(X_train, X_test, Y_train)   # Logistic Regression
elif CLASSIFIER == 'BNB':
    model = get_BNB(X_train, X_test, Y_train)   # Bernoulli Naive Bayes
elif CLASSIFIER == 'RF':
    model = get_RF(X_train, X_test, Y_train)   # Random Forest Classifier
elif CLASSIFIER == 'KNN':
    model = get_KNN(X_train, X_test, Y_train, n=5)
elif CLASSIFIER == 'XGB':
    model = get_XGB(X_train, X_test, Y_train)
elif CLASSIFIER == 'DT':
    model = get_DT(X_train, X_test, Y_train)


# predict values for evaluation model and create submission file
print("Predicting for test data.")
eval_probs = model.predict_proba(X_eval)
# picking best 7 choices
recs = (-eval_probs).argsort()[:, :7]

# outputing the submission file
output = output_submission(df_eval, eval_probs, recs, labels)

# getting MAP metrics
map_score = get_MAP(np.array(Y_test), model.predict_proba(X_test))
print(map_score)

