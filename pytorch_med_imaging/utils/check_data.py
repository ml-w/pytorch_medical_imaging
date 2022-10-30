import pandas as pd
import re
from pytorch_med_imaging.Algorithms.utils import get_unique_IDs
from typing import Union, List, Iterable, Optional
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


