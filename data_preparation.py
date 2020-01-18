def data_processing():
    """Manipulation of Breast Cancer Wisconsin (Diagnostic) Data Set.

    Initially, 699 instances (patients) with 9 features each.
    In the initial data set, a tumor is classified as benign (corresponding to number 2 at the last column)
    or malignant (corresponding to number 4 at the last column).

    Returns normalized training and validation sets.
    """

    from sklearn import preprocessing
    import pandas as pd

    #Read the file containing the data.
    df = pd.read_csv('wisconsin-cancer-dataset.csv',header=None)

    #Process the data.

    #There are some missing data in column 6. We exclude these rows.
    #There are 683 remaining rows with all the features included.
    df = df[~df[6].isin(['?'])]

    #In the data, last column (i.e. 10) is 2 for benign and 4 for malignant.
    #Change it to be 0 for benign and 1 for malignant.
    df.iloc[:,10].replace(2, 0,inplace=True)
    df.iloc[:,10].replace(4, 1,inplace=True)

    #Data normalization.
    names = df.columns[1:10]
    scaler = preprocessing.MinMaxScaler()
    scaled_df = scaler.fit_transform(df.iloc[:, 1:10])
    scaled_df = pd.DataFrame(scaled_df, columns=names)

    #Split data set into training and validation sets.
    #Training set (first 500 instances).
    x_train=scaled_df.iloc[0:500,:].values.transpose()
    y_train=df.iloc[0:500,10:].values.transpose()
    #Validation set.
    x_val=scaled_df.iloc[501:683,:].values.transpose()
    y_val=df.iloc[501:683,10:].values.transpose()

    return x_train, y_train, x_val, y_val