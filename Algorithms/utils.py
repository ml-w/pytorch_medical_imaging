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


def get_fnames_by_IDs(fnames, idlist, globber=None):
    if globber is None:
        globber = "(?=.*%s.*)"

    outfnames = {}
    for id in idlist:
        flist = []
        for f in fnames:
            if not re.match(globber%id, f) is None:
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






