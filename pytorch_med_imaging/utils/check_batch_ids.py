import configparser
import pandas as pd
import numpy
import itertools
from pathlib import Path

def read_ini(f_path):
    cfg = configparser.ConfigParser()
    cfg.read(f_path)
    return cfg['FileList']['training'].split(','), \
           cfg['FileList']['testing'].split(',')

def create_batch_series(all_ids, check_ids, name):
    return pd.Series(data=[_id in all_ids for _id in check_ids], index=check_ids, name=name)

def main():
    batch_data = {}
    ini_files = Path('.').glob('*.ini')
    for f in ini_files:
        batch_name = f.name.replace('.ini', '')
        ids = read_ini(str(f))
        batch_data[batch_name] = ids
        if len(list(set(ids[0]))) != len(ids[0]):
            raise IndexError(f"Some ids are duplicates for {f} training!")
        if len(list(set(ids[1]))) != len(ids[1]):
            raise IndexError(f"Some ids are duplicates for {f} testing!")

    validation_ids = [r.rstrip() for r in open('./Validation.txt', 'r').readlines()]
    if len(list(set(validation_ids))) != len(validation_ids):
        raise IndexError(f"Some ids are duplicates for validation!")

    # creat check table
    all_ids = set.union(*([set.union(set(a), set(b)) for a, b, in batch_data.values()] + [set(validation_ids)]))
    cols = []
    for key, (a, b) in batch_data.items():
        name_a, name_b = (key, 'training'), (key, 'testing')
        sa = create_batch_series(all_ids, a, name_a)
        sb = create_batch_series(all_ids, b, name_b)
        cols.extend([sa, sb])
    cols.append(create_batch_series(all_ids, validation_ids, ('validation', '')))
    check_table = pd.concat(cols, axis=1).sort_index()
    check_table.index = check_table.index.astype('str')

    # check overlap between testing sets and validation
    for a, b in itertools.combinations(check_table.columns, 2):
        if a[1] == 'training' or b[1] == 'training':
            continue
        seta = set(check_table[a][check_table[a] == True].index)
        setb = set(check_table[b][check_table[b] == True].index)
        if len(list(seta.intersection(setb))) != 0:
            print(f"WARNING! Overlap between {a} and {b}: {seta.intersection(setb)}")

    # create a table which record which ID is in testing set of which batch
    folds = pd.Series(dtype=str, name="Fold")
    for ids in all_ids:
        ss = check_table.fillna(False).loc[ids][:, 'testing']
        try:
            fold = ss.index[ss] # This gives an index object
            folds[ids] = fold[0]
        except:
            if ids in validation_ids:
                folds[ids] = 'validation'
            else:
                raise ArithmeticError(f"No group was assigned to {ids}")
    folds.sort_index(inplace=True)

    # counts
    summary = check_table.count(axis=0)

    # write to excel output
    writer = pd.ExcelWriter('./summary.xlsx')
    summary.to_frame().to_excel(writer, sheet_name="Summary")
    check_table.fillna(False).to_excel(writer, sheet_name="Check List")
    folds.to_frame().to_excel(writer, sheet_name="Fold Config")
    writer.save()
    writer.close()

if __name__ == '__main__':
    main()
