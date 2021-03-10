import re, os
from pytorch_med_imaging.logger import Logger

__all__ = ['get_unique_IDs', 'get_fnames_by_globber', 'get_fnames_by_IDs', 'load_supervised_pair_by_IDs']



def get_unique_IDs(fnames, globber=None):
    idlist = []
    for f in fnames:
        if globber is None:
            globber = "([0-9]{3,5})"

        mo = re.search(globber, f)
        if not mo is None:
            idlist.append(f[mo.start():mo.end()])

    idlist = list(set(idlist))
    idlist.sort()
    return idlist


def get_fnames_by_IDs(fnames, idlist, globber=None):
    _logger = Logger['algorithm.utils']
    if globber is None:
        globber = "([0-9]{3,5})"

    outfnames = {}
    for id in idlist:
        flist = []
        for f in fnames:
            _f = os.path.basename(f)
            l = re.findall(globber, _f)
            if not len(l):
                continue
            if l[0] == id:
                flist.append(f)
        # skip if none is found
        if len(flist) == 0:
            _logger.warning(f"Can't found anything for key {id}. Skipping..")
            continue
        outfnames[id] = flist
    return outfnames


def get_fnames_by_globber(fnames, globber):
    assert isinstance(fnames, list)

    copy = list(fnames)
    for f in fnames:
        if re.match(globber, f) is None:
            copy.remove(f)
    return copy


def load_supervised_pair_by_IDs(source_dir, target_dir, idlist, globber=None):
    source_list = get_fnames_by_globber(os.listdir(source_dir), globber) \
        if not globber is None else os.listdir(source_dir)
    _logger = Logger['algorithm.utils']

    source_list = get_fnames_by_IDs(source_list, idlist)
    source_keys = source_list.keys()
    source_list = [source_list[key][0] for key in source_list]
    target_list = get_fnames_by_IDs(os.listdir(target_dir), idlist)
    target_keys = target_list.keys()
    target_list = [target_list[key][0] for key in source_keys]

    if len(source_list) != len(target_list):
        _logger.error("Dimension mismatch when pairing.")
        missing = {'Src': [], 'Target': []}
        for src in source_keys:
            if src not in target_keys:
                missing['Src'].append(src)
        for tar in target_keys:
            if tar not in source_keys:
                missing['Target'].append(src)
        _logger.debug(f"{missing}")
        raise ValueError("Dimension mismatch! Src: %i vs Target: %i"%(len(source_list), len(target_list)))

    return source_list, target_list

def directory_sorter(dir, sort_dict=None, pre_filter=None):
    import fnmatch
    import shutil
    all_nii_files = os.listdir(dir)
    all_nii_files.sort()
    all_nii_files = fnmatch.filter(all_nii_files,'*nii.gz')

    if sort_dict is None:
        sort_dict = {'T2WFS':   "(?i)(?=.*T2.*)(?=.*(fs|stir).*)",
                     'T2W':     "(?i)(?=.*T2.*)(?!.*(fs|stir).*)",
                     'CE-T1WFS':"(?i)(?=.*T1.*)(?=.*\+[cC].*)(?=.*(fs|stir).*)",
                     'CE-T1W':  "(?i)(?=.*T1.*)(?=.*\+[cC].*)(?!.*(fs|stir).*)",
                     'T1W':     "(?i)(?=.*T1.*)(?!.*\+[cC].*)(?!.*(fs|stir).*)"
                 }

    if pre_filter is None:
        pre_filter = {'SURVEY': "(?i)(?=.*survey.*)",
                      'NECK': "(?i)(?=.*neck.*)"}

    directions = {'_COR': "(?i)(?=.*cor.*)",
                  '_TRA': "(?i)(?=.*tra.*)",
                  '_SAG': "(?i)(?=.*sag.*)",}

    for p in pre_filter:
        if not os.path.isdir(dir + '/' + p):
                os.makedirs(dir + '/' + p, exist_ok=True)

        remove=[]
        for ff in all_nii_files:
            if not re.search(pre_filter[p], ff) is None:
                try:
                    shutil.move(dir + '/' + ff, dir + '/' + p)
                except Exception as e:
                    print(e.args[0])
                    print(os.path.join(dir, ff), os.path.join(dir, p))
                remove.append(ff)

        for r in remove:
            all_nii_files.remove(r)


    for d in directions:
        for f in sort_dict:
            if not os.path.isdir(dir + '/' + f + d):
                os.mkdir(dir + '/' + f + d)

            remove=[]
            for ff in all_nii_files:
                if not re.search(sort_dict[f] + directions[d], ff) is None:
                    try:
                        shutil.move(dir + '/' + ff, dir + '/' + f + d)
                    except Exception as e:
                        print(e.args[0])
                        print(os.path.join(dir, ff), os.path.join(dir, f + d))
                    remove.append(ff)

            for r in remove:
                all_nii_files.remove(r)


    # move all unhandled to misc folder
    if not os.path.isdir(dir + '/misc'):
        os.mkdir(dir + '/misc')

    for ff in all_nii_files:
        try:
            shutil.move(dir + '/' + ff, dir + '/misc')
        except Exception as e:
            print(e.args[0])


def directory_index(dir, out_csv, id_globber="(^[a-zA-Z0-9]+)"):

    import pandas as pd

    # Folders name will be their categories
    folders = os.listdir(dir)

    idset = {}
    for f in folders:
        if not f in idset:
            idset[f] = []

        files = os.listdir(os.path.join(dir, f))
        for ff in files:
            if not ff.endswith('.gz') and not ff.endswith('.nii'):
                continue
            fid = re.search(id_globber, ff)
            if not fid is None:
                idset[f].append(fid.group())

    # Obtain a list of all ids
    allids = [idset[a] for a in idset]
    allids = [b for a in allids for b in a]
    allids = list(set(allids))

    outdf = pd.DataFrame()
    for ids in allids:
        row = [ids] + ["YES" if ids in idset[key] else "NO" for key in idset]
        col = ['Study Number'] + list(idset.keys())

        row = pd.DataFrame([row], columns=col)
        outdf = outdf.append(row)

    outdf.to_csv(out_csv, index=False)


if __name__ == '__main__':
    # directory_sorter('../NPC_Segmentation/0A.NIFTI_ALL/Malignant')
    directory_index('../NPC_Segmentation/0A.NIFTI_ALL/Malignant', '/home/lwong/FTP/temp/images.csv')




