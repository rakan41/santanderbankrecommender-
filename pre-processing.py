"""
Kaggle Competition: Santander Product Recommendation
Authors:
Rakan Frijat
Ian Commerford

Data source:
https://www.kaggle.com/c/santander-product-recommendation/data
The script assumes that the zip file has been downloaded into the same directory

Python Version: 3.6

Program Arguments Taken
Argument 1: directory to read and write data to/from
Argument 2: name of training file
Argument 3: name of evaluation file
Argument 4: name of the source zip file
Sample argument: data/ train_ver2.csv test_ver2.csv santander-product-recommendation.zip

"""
# Import Modules
import sys
import pandas as pd
import os
from zipfile import ZipFile

# Program Arguments
args = sys.argv
filepath = args[1]
training_file = args[2]
test_file = args[3]
downloaded_zip_file = args[4]

# Parameters
LAG_PERIOD = 3      # specifies number of preceding periods are required to generate lag variables

# define the array of products
labels = ['ind_ahor_fin_ult1', 'ind_aval_fin_ult1', 'ind_cco_fin_ult1', 'ind_cder_fin_ult1',
          'ind_cno_fin_ult1', 'ind_ctju_fin_ult1', 'ind_ctma_fin_ult1', 'ind_ctop_fin_ult1',
          'ind_ctpp_fin_ult1', 'ind_deco_fin_ult1', 'ind_deme_fin_ult1', 'ind_dela_fin_ult1', 'ind_ecue_fin_ult1',
          'ind_fond_fin_ult1', 'ind_hip_fin_ult1', 'ind_plan_fin_ult1', 'ind_pres_fin_ult1', 'ind_reca_fin_ult1',
          'ind_tjcr_fin_ult1', 'ind_valo_fin_ult1', 'ind_viv_fin_ult1', 'ind_nomina_ult1', 'ind_nom_pens_ult1',
          'ind_recibo_ult1']

# define string variables to convert to categorical
string_cols = ['sex', 'employee_flag', 'country_resi', 'new_cust_flag', 'prim_cust', 'cust_type',
               'cust_relation_type', 'resi_flag', 'foreigner_flag', 'spouse_of_employee_flag',
               'cust_channel', 'deceased_flag', 'addr_type', 'province_code', 'province', 'active_flag',
               'cust_segment']


# takes a large CSV and splits it into smaller files based on the partition date
def split_file(source_file):
    print("Splitting dataset into partitions:")

    # creates output directory if not exists
    if not os.path.exists('{}partitionedfiles'.format(filepath)):
        os.makedirs('{}partitionedfiles'.format(filepath))

    # opens source file and reads it line by line
    partition_dates = []
    with open('{}{}'.format(filepath, source_file), 'r') as f_in:
        # read firstline to get header
        header_line = f_in.readline()
        # loop through remaining lines and create output files for each date
        current_date = None
        f_out = None

        for line in f_in:
            new_date = line.split(',')[0]
            # if we've reached a new date partition in the data
            if new_date != current_date:
                # close output file
                if f_out:
                    f_out.close()
                # create a new output file for the new date
                print("Splitting partition {}...".format(new_date), end="\n")
                partition_dates.append(new_date)
                f_out = open('{}partitionedfiles/{}.csv'.format(filepath, new_date), 'w')
                f_out.write(header_line)
                current_date = new_date
            # append current line to the output file
            f_out.write(line)
        f_out.close()
        print("Splitting complete.", end="\n")

        # return list of available partition dates
        return partition_dates


# formats partitioned csvs and then saves them to pickle.
# transformations includes renaming spanish column names to english, converting datatypes, filling NA values
# arguments
# partitions: list of partitioned csvs to process
# filetype: use 'train' for training datasets and 'test' for test
# removecsv: mark as true if you want the csv partitions from previous function to be deleted after generating a pickled partition
#
def format_partitions(partitions, filetype='train', removecsv=True):
    print("Formatting partitions:")

    # create new formatted pickle file for each partition
    for part_dt in partitions:
        print("Creating pickle file for partition date {}...".format(part_dt), end=" ")

        # load partition csv to dataframe
        df = pd.read_csv('{}partitionedfiles/{}.csv'.format(filepath, part_dt), low_memory=False)

        # rename column headers to English (except for product names)
        df = df.rename(index=str, columns={"fecha_dato": "part_dt", "ncodpers": "customer_code",
                                           "ind_empleado": "employee_flag", "pais_residencia": "country_resi",
                                           "sexo": "sex",
                                           "fecha_alta": "cust_first_date", "ind_nuevo": "new_cust_flag",
                                           "antiguedad": "customer_duration", "indrel": "prim_cust",
                                           "ult_fec_cli_1t": "cust_end_date", "indrel_1mes": "cust_type",
                                           "tiprel_1mes": "cust_relation_type", "indresi": "resi_flag",
                                           "indext": "foreigner_flag",
                                           "conyuemp": "spouse_of_employee_flag",
                                           "canal_entrada": "cust_channel", "indfall": "deceased_flag",
                                           "tipodom": "addr_type", "cod_prov": "province_code",
                                           "nomprov": "province", "ind_actividad_cliente": "active_flag",
                                           "renta": "cust_income",
                                           "segmento": "cust_segment"})

        # cast product columns as booleans
        if filetype == 'train':
            df[labels] = df[labels].astype('bool')

        # replace null values with mean for continuous variables
        continuous_cols = ['age', 'cust_income', 'customer_duration']
        df[continuous_cols] = df[continuous_cols].apply(pd.to_numeric, errors='coerce')
        df[continuous_cols] = df[continuous_cols].fillna(df[continuous_cols].mean(numeric_only=True))

        # casting date variables as datetime
        date_cols = ['cust_end_date', 'cust_first_date', 'part_dt']
        for col in date_cols:
            df[col] = pd.to_datetime(df[col])

        # cast string columns as string
        df[string_cols] = df[string_cols].astype(str)

        # clean up nan values
        df[string_cols] = df[string_cols].fillna("NA")

        # clean free text category columns
        free_text_cols = ['cust_type', 'new_cust_flag']
        for col in free_text_cols:
            df[col] = df[col].str.replace(".0", "")

        # cast string columns into categorical types
        df[string_cols] = df[string_cols].astype('category')

        # write formatted partition to pickle
        df.to_pickle("{}partitionedfiles/{}.pickle.zip".format(filepath, part_dt), compression='zip')

        # remove staging partition csv
        if removecsv == True:
            os.remove("{}partitionedfiles/{}.csv".format(filepath, part_dt))

        print("completed.")

    print("Formatting complete.")
    return


# applies various transforrmations to a list of partition files and returns transformed files. Transformations include
#  creating summary variables, lag variables remapping categorical variables, creating output labels,
#  and indexing the data by customer and part_dt.
# arguments
# partitions: an array of the partitions to transform. The list must be at least lag periods + 1
# filetype: use 'train' for training datasets and 'test' for test
def transform_partition(partitions, filetype='train'):
    print("Transforming partitions.")
    # creates output directory if not exists
    if not os.path.exists('{}transformedfiles'.format(filepath)):
        os.makedirs('{}transformedfiles'.format(filepath))

    # catch any partition lists that do not have enough partitions for the required number of lag periods
    if len(partitions) < LAG_PERIOD + 1:
        print("Error. Not enough partitions for the number of lag periods.")
        return

    partitions.sort()

    df = pd.read_pickle("{}partitionedfiles/{}.pickle.zip".format(filepath, partitions[0]), compression='zip')


    for i in range(len(partitions)-1):
        print("Merging {} with {}".format(partitions[i], partitions[i+1]))
        df2 = pd.read_pickle("{}partitionedfiles/{}.pickle.zip".format(filepath, partitions[i + 1]), compression='zip')
        # creates empty labels so that DFs can be union'ed without errors
        if filetype == 'test' and i == len(partitions)-1-1:
            for l in labels:
                df2[l] = False
        df = pd.concat([df, df2], axis=0)

    print("Merging complete.")
    print("Setting primary index as customer code, secondary index as partititon date.")
    # set indices first by partition date, then customer code
    df.set_index(['customer_code', 'part_dt'], inplace=True)
    df.sort_index(inplace=True)

    print("Creating labels.")
    # create output labels of whether a customer bought a product next month
    for i in range(len(labels)):
        new_label = labels[i] + "_new"
        df[new_label] = False
        df.loc[((df[labels[i]].shift(1) == False) & (df[labels[i]] == True)), new_label] = True

    print("Generating lag variables.")

    # create lag variables for products added
    for i in range(1, LAG_PERIOD+1):
        df['total_added_t_{}'.format(str(i))] = ((df[labels].shift(i+1) == False) & (df[labels].shift(i) == True)).sum(axis=1)

    # create lag variable for current products held.
    for i in range(1, LAG_PERIOD+1):
        df[[label + "_t_{}".format(str(i)) for label in labels]] = df[labels].shift(i, fill_value=False)

    # create a column that counts total products held by customer in previous period
    df['total_products_t_1'] = df[labels].shift(1).sum(axis=1)

    # drop products held in current partition. this is because this won't be available in test data
    df.drop(labels, axis=1, inplace=True)

    # recast categorical variables (sometimees their datatype reverts to obkect)
    print("Recasting categorical variables")
    df[string_cols] = df[string_cols].astype('category')

    # save pickle files for each partition that has a full set of lag data
    for part in partitions[LAG_PERIOD:]:
        print("Writing {} to pickle.".format(part))
        df.loc[(slice(None), part), :].to_pickle("{}transformedfiles/{}.pickle.zip".format(filepath, part), compression='zip')

    return df


def unzip_data(file_name, destination_dir):
    with ZipFile(file_name, 'r') as z:
        z.extractall(destination_dir)


# calling pre-processing functions (comment out any steps you want to skip)
unzip_data(downloaded_zip_file, filepath)
training_partitions = split_file(training_file)
format_partitions(training_partitions, filetype='train')
transform_partition(training_partitions)
test_partition = split_file(test_file)
format_partitions(test_partition, filetype='test')
test_partitions = training_partitions[-LAG_PERIOD:] + test_partition
transform_partition(test_partitions, filetype='test')

df = pd.read_pickle("{}transformedfiles/{}.pickle.zip".format(filepath, "2016-05-28"))




