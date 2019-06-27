import re
import os
import fnmatch

def get_unique_IDs(files, globber=None):
    if globber is None:
        globber = "[^T][0-9]+"
    outlist = []
    for f in files:
        matchobj = re.search(globber, f)
        if not matchobj is None:
            outlist.append(int(f[matchobj.start():matchobj.end()]))
    return list(set(outlist))

def get_in_a_not_b(a, b):
    out = []
    for aa in a:
        if not aa in b:
            out.append(aa)
    return out


if __name__ == '__main__':
    fold_1 = get_unique_IDs(os.listdir('../NPC_Segmentation/98.Output/KFold_Perfect_000_RESULT'))
    fold_2 = get_unique_IDs(os.listdir('../NPC_Segmentation/98.Output/KFold_Perfect_001_RESULT'))
    fold_1.sort()
    fold_2.sort()
    gt = get_unique_IDs(os.listdir('../NPC_Segmentation/21.NPC_Perfect_SegT2/00.First'))

    # print get_in_a_not_b(fold_1, gt)
    # print get_in_a_not_b(gt, fold_1)
    import pandas as pd
    ef = pd.ExcelFile("/home/lwong/FTP/temp/data_npc_analysis.xlsx")
    pd = ef.parse("Perfect-4-Fold")
    pd = pd.sort_values('Index')


    # print list(pd[pd['Fold']==1]['Index'])
    print get_in_a_not_b(fold_1, list(pd[pd['Fold']== 1]['Index']))
    print get_in_a_not_b(fold_2, list(pd[pd['Fold']== 2]['Index']))
