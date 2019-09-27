import re, os

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


def get_fnames_by_IDs(fnames, idlist):
    outfnames = {}
    for id in idlist:
        flist = []
        for f in fnames:
            if not re.match("(?=.*%s.*)"%(id), f) is None:
                flist.append(f)
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
    source_list = get_fnames_by_IDs(source_list, idlist)
    source_list = [source_list[key][0] for key in source_list]
    target_list = get_fnames_by_IDs(os.listdir(target_dir), idlist)
    target_list = [target_list[key][0] for key in target_list]

    if len(source_list) != len(target_list):
        raise ValueError("Dimension mismatch! Src: %i vs Target: %i"%(len(source_list), len(target_list)))

    return source_list, target_list

def directory_sorter(dir, sort_dict=None):
    import fnmatch
    import shutil
    all_nii_files = os.listdir(dir)
    all_nii_files.sort()
    all_nii_files = fnmatch.filter(all_nii_files,'*nii.gz')

    if sort_dict is None:
        sort_dict = {'T2WFS':"(?=.*T2.*)(?=.*[fF][sS].*)",
                     'T2W': "(?=.*T2.*)(?!.*[fF][sS].*)",
                     'CE-T1WFS': "(?=.*T1.*)(?=.*\+[cC].*)(?=.*[fF][sS].*)",
                     'CE-T1W': "(?=.*T1.*)(?=.*\+[cC].*)(?!.*[fF][sS].*)",
                     'T1W': "(?=.*T1.*)(?!.*\+[cC].*)(?!.*[fF][sS].*)"
                 }

    directions = {'_COR': "(?i)(?=.*cor.*)",
                  '_TRA': "(?i)(?=.*tra.*)",
                  '_SAG': "(?i)(?=.*sag.*)",}

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

if __name__ == '__main__':
    directory_sorter('/home/lwong/FTP/2.Projects/8.NPC_Segmentation/0A.NIFTI_ALL/')





