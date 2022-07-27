import argparse
import re
import SimpleITK as sitk
import pandas as pd
from pathlib import Path
from tqdm import tqdm


def main(*args, **kwargs):
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', action='store',
                        help="Director to process.")
    parser.add_argument('-o', '--output',
                        help="Output path of a csv file that contains the list of IDs and its path where "
                             "the volume is zero.")
    parser.add_argument('--idGlobber', action='store', default="^[^\W]+",
                        help="Glob IDs from the name.")
    a = parser.parse_args()

    in_path = Path(a.input)
    out_path = Path(a.output).with_suffix('.csv')

    # in case there are .nii and not .nii.gz files
    rows = []
    for f in tqdm(list(in_path.glob("*nii???"))):
        id = re.search(a.idGlobber, str(f.name)).group()
        vol = get_volume(f)
        rows.append(pd.Series(data=[vol], index=[id], name='Volume (cm^3)'))

    df = pd.concat(rows)
    df.index = df.index.astype(str)
    if out_path.is_file(): # update if exist
        _df = pd.read_csv(str(out_path), index_col=0)
        _df.index = _df.index.astype(str)
        overlap = _df.index.intersection(df.index)
        new = set(df.index) - set(_df.index)
        _df.update(df.loc[overlap], index_col=0)
        df = _df.join(df.loc[new], axis=1)
        print(f"Updated rows: {new}")
    df.sort_index(inplace=True)
    df.to_csv(str(out_path))

def get_volume(f: Path) -> float:
    r"""Return volume in cm^3 unit"""
    seg_im = sitk.ReadImage(str(f), sitk.sitkUInt8)
    stat_filter = sitk.LabelShapeStatisticsImageFilter()
    stat_filter.Execute(seg_im != 0)
    try:
        size_mm3 = stat_filter.GetPhysicalSize(1)
    except RuntimeError:
        size_mm3 = 0
    size_cm3 = size_mm3 / 1000.
    return size_cm3

if __name__ == '__main__':
    main()