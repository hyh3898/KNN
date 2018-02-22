import pandas as pd
import numpy as np
import sys


# Inputed the '?' in the dataset


def process_nu(df):
    nu_columns = [1,2,7,10,13,14]
    df[nu_columns] = df[nu_columns].replace('?', np.NaN)
    df[nu_columns] = df[nu_columns].apply(pd.to_numeric)

    means = df[nu_columns].mean()
    for col in means.index:
        df[col] = df[col].replace(np.NaN, means[col])
    return df

def process_bi(df):
    bi_columns = [0,3,4,5,6,8,9,11,12]

    for i in bi_columns:
        freqs = df[i].value_counts()
        df[i] = df[i].replace('?', freqs.index[0])

        c = list(df[i].value_counts().index)
        n = list(range(df[i].value_counts().shape[0]))
        dict_ = dict(zip(c,n))

        for k,v in df[i].iteritems():
            df.loc[k,i] = dict_[v]
    return df

def Z_norm(df):
    length = df.shape[1]
    for i in range(length-1):
        df[i] = (df[i]-df[i].mean())/df[i].std()

    return df

def preprocess(df):
    df = process_nu(df)
    df = process_bi(df)
    df = Z_norm(df)
    return df

def main():
    # output to file
    if len(sys.argv) > 1:
        script_name = sys.argv[0]
        infile = sys.argv[1]
        df = pd.read_csv(infile, header=None)
        df = preprocess(df)
        if infile == 'crx.data.training':
            outfile = 'crx.training.processed'
        elif infile == 'crx.data.testing':
            outfile = 'crx.testing.processed'
        df.to_csv(outfile, header=None, index=None)
    else:
        print('no file being processed')


main()
