import pandas as pd
import re
from sklearn import model_selection
import os
from numpy import random

def GenerateFileList(files):
    regex_dict = {'T1W':        "((?=.*T1.*)(?!.*FS.*)(?!.*[cC].*))",
                  'CE-T1W':     "((?=.*T1.*)(?!.*FS.*)(?=.*[cC].*))",
                  'CE-T1W-FS':  "((?=.*T1.*)(?=.*FS.*)(?=.*[cC].*))",
                  'T2W-FS':     "((?=.*T2.*)(?=.*FS.*))"}

    # get unique indexes
    globber = "[0-9]+"
    outlist = []
    for f in files:
        matchobj = re.search(globber, f)
        if not matchobj is None:
            outlist.append(int(f[matchobj.start():matchobj.end()]))
    idlist = list(set(outlist))

    df = pd.DataFrame()
    for i in idlist:
        row = {'None': []}
        fs = fnmatch.filter(files, str(i) + '*')
        for ff in fs:
            FLAG = False
            for k in regex_dict:
                matchobj = re.match(regex_dict[k], ff)
                if not matchobj is None and k not in row:
                    row[k] = ff
                    FLAG = True
            if not FLAG:
                row['None'].append(ff)

        if len(row['None']) == 0:
            row['None'] = "NULL"
        else:
            row['None'] = '|'.join(row['None'])

        tmp = pd.DataFrame([list(row.values())], columns=list(row.keys()), index=[i])
        df = df.append(tmp)
    return df[list(regex_dict.keys()) + ['None']]



def GenerateTestBatch(ids, k_fold, outdir, prefix="Batch_", exclude_list=None, stratification_class=None, validation=0):
    import configparser
    try:
        ids = list(ids.tolist())
    except:
        pass

    if not os.path.isdir(outdir):
        os.makedirs(outdir, exist_ok=True)

    # if validation required
    if validation > 0:
        if stratification_class is None:
            validation_ids = random.choice(ids, size=validation, replace=False)
            for v in validation_ids:
                ids.remove(v)
        else:
            classes = list(set(stratification_class))
            classes_prob = {c: stratification_class.count(c) / float(len(ids)) for c in classes}
            p = [classes_prob[s] / float(stratification_class.count(s)) for s in stratification_class]
            print(len(p), len(ids))
            validation_ids = random.choice(ids, size=validation, p=p, replace=False)


            tmp_stras = {i: stra for i, stra in zip(ids, stratification_class)}
            for v in validation_ids:
                tmp_stras.pop(v)
            ids = list(tmp_stras.keys())
            stratification_class = [tmp_stras[k] for k in tmp_stras]

        val_file = open(os.path.join(outdir, 'Validation.txt'), 'w')
        val_file.writelines([str(v) + '\n' for v in validation_ids])


    # Set up output dictionary for writing
    out = {}

    # Clean excluded list from input ids
    if not exclude_list is None:
        if isinstance(exclude_list, str):
            exclude_list = [r.rstrip() for r in open(exclude_list, 'r').readlines()]

        if isinstance(exclude_list, list):
            for e in exclude_list:
                if e in ids:
                    ids.remove(e)
                    print("Removed ", e)
                else:
                    print(e, " not in list")

    # Determine if stratified sampling is used for class balances
    if stratification_class is None:
        splitter = model_selection.KFold(n_splits=k_fold, shuffle=True)
        get_split = lambda x: splitter.split(x)
    else:
        splitter = model_selection.StratifiedKFold(n_splits=k_fold, shuffle=True)
        get_split = lambda x: splitter.split(x, stratification_class)

    # Create folder if not exist
    os.makedirs(outdir, exist_ok=True)
    # Determine train test fold split
    for i, (train_index, test_index) in enumerate(get_split(ids)):
        train_ids = [ids[i] for i in train_index]
        test_ids = [ids[i] for i in test_index]
        train_ids.sort()
        test_ids.sort()
        train_ids = [str(x) for x in train_ids]
        test_ids = [str(x) for x in test_ids]
        print("Train: %d, Test: %d"%(len(train_ids), len(test_ids)))
        out[i] = {'train_id': train_ids, 'test_ids': test_ids}



    # Output files
    for k in range(k_fold):
        outfile = os.path.join(outdir, prefix + '%02d.ini'%k)

        outconf = configparser.ConfigParser()
        outconf['FileList'] = {'testing': ','.join(out[k]['test_ids']), 'training': ','.join(out[k]['train_id'])}
        with open(outfile, 'w') as f:
            outconf.write(f)


def check_batches_files(dir, globber=None):
    files = os.listdir(dir)

    # get list of each batch
    b = []
    d = {}
    for f in files:
        mo = re.search(globber if not globber is None else '[0-9]{3}_.*ing', f)
        if not mo is None:
            batch = f[mo.start():mo.end()]
            d[batch] = [r.rstrip() for r in open(dir + '/' + f, 'r').readlines()]
            d[batch].sort()
            b.append(batch[:3])
    b = list(set(b))

    for bb in b:
        # check if there are test/train overlap
        train = d[bb + '_Training']
        test = d[bb + '_Testing']

        intersection = list(set(train) & set(test))
        if len(intersection):
            print("Intersection: ", bb, intersection)


    # check if there are repeated number in each list
    for k in d:
        if len(list(set(d[k]))) != len(d[k]):
            print("Repated number: ",k)


    # check if there are inter-batch overlap
    for bb in b:
        for cc in b:
            if bb == cc:
                continue
            test_b = d[bb + '_Testing']
            test_c = d[cc + '_Testing']

            inter_test = list(set(test_c) & set(test_b))
            if len(inter_test):
                print("Batch overlap!", bb, cc, inter_test)

    # check if all folds combine to give the same set list
    fulllist = {}
    for bb in b:
        train = d[bb + '_Training']
        test = d[bb + '_Testing']

        fulllist[bb] = list(set(train) & set(test))

    for k1 in fulllist:
        for k2 in fulllist:
            if k1 == k2:
                continue
            if fulllist[k1] != fulllist[k2]:
                print("Difference in full: ", k1, k2)


    pass


if __name__ == '__main__':
    import pandas as pd
    allids = [r.rstrip() for r in open('../../NPC_Segmentation/99.Testing/Survival_analysis/chemo_only.txt').readlines()]
    datasheet = pd.read_csv('../../NPC_Segmentation/50.NPC_SurvivalAnalysis/survival_analysis_data_table.csv',
                            index_col=0)
    datasheet.index = datasheet.index.astype(str)
    datasheet=datasheet.loc[allids]
    dfs = (datasheet['DFS']*12).astype('int').to_list()

    GenerateTestBatch(allids,
                      5,
                      '../../NPC_Segmentation/99.Testing/Survival_5Fold_Chemo_only_stratified_DFS/',
                      stratification_class=dfs,
                      validation=36,
                      prefix='B'
                      )

    # print GenerateFileList(os.listdir('../NPC_Segmentation/06.NPC_Perfect')).to_csv('~/FTP/temp/perfect_file_list.csv')

    # check_batches_files('../NPC_Segmentation/99.Testing/B06/')

