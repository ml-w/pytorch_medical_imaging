import os
import pandas as pd

def concat_df(*args):
    assert all([isinstance(a, pd.DataFrame) for a in args]), "Some input are not dataframes."

    df = pd.DataFrame()
    for frame in args:
        df = df.append(frame)

    df = df.sort_index()
    return df

def concat_csv(*args):
    assert all([os.path.isfile(a) for a in args]), "Some of the input cannot be found."

    dfs = [pd.read_csv(a, index_col='IDs') for a in args]
    return concat_df(*dfs)

def concat_csv_recursive_search(d):

    target_csv = []
    for root, dirs, files in os.walk(d):
        if len(files) > 0:
            for f in files:
                if not f.endswith('.csv'):
                    continue
                else:
                    print("Adding file: ", os.path.join(root, f))
                    target_csv.append(os.path.join(root, f))
    target_csv.sort()
    return target_csv

if __name__ == '__main__':
    csvs = concat_csv_recursive_search("../NPC_Segmentation/98.Output/BM")
    df = concat_csv(*csvs)
    df.to_csv('/home/lwong/FTP/temp/BM/bm_result_3fold(2).csv')


