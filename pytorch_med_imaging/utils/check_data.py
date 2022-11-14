import pandas as pd
from pytorch_med_imaging.utils.uid_ops import get_unique_IDs
from typing import Union, Optional
from pathlib import Path

def check_image_exist(dataids: Union[list, set],
                      check_dir: Union[Path, str],
                      idGlobber: Optional[str] = "^\w+\d+"):
    r"""Check if images fitting the ids listed can be found within the check_dir.

    Args:
        dataids (list):
            List of unique ids to check
        check_dir (Path or str):
            Directory to check.
    """

    check_dir = Path(check_dir)
    if len(dataids) != len(set(dataids)):
        raise RuntimeWarning("Dataids are not unique.")

    existing_ids = get_unique_IDs([str(r.name) for r in check_dir.glob("*nii.gz")], idGlobber)

    missing_in_dir = set(dataids) - set(existing_ids)
    return missing_in_dir

if __name__ == '__main__':
    datasheet = Path("../../NPC_Segmentation/99.Testing/BM_ISMRM/ISMRM_datasheet_v6.csv")
    datasheet = pd.read_csv(datasheet, index_col=0)
    ids = datasheet.index.to_list()

    targetdir = Path("../../NPC_Segmentation/60.Large-Study/v1-All-Data/Normalized_2/T2WFS_TRA/01.NyulNormalized")

    # report checking results
    missing_ids = check_image_exist(ids, targetdir)
    print(f"{len(missing_ids)} were missing: {missing_ids}")

    # remove non-existin gones from the list and save to a new datasheet
    new_df = datasheet.drop(missing_ids)
    new_df.to_csv("../../NPC_Segmentation/99.Testing/BM_ISMRM/ISMRM_datasheet_v6.csv")

