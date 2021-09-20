import yaml
import torchio as tio
from pathlib import Path

__all__ = ['create_transform_compose']

def create_transform_compose(yaml_file: Path) -> tio.Compose:
    r"""
    Read yaml file that allows you to create a `tio.Compose` object for use to augment/pre-process the input images

    Args:
        yaml_file (str or Path):
            The yaml file.

    Returns:
        tio.Compose

    Example yaml file:
    ::
        ToCanonical:
          {} # Means there are no arguments.

        RandomAffine:
          scales: [0.9, 1.1]
          degrees: 10

        RandomFlip:
          - 'lr'

        RandomNoise:
          std: [0, 8]

        RandomRescale:
          - 500
          - [1, 255]
          - 'corner-pixel'
        ```
    """
    yaml_file = Path(yaml_file)
    if not yaml_file.exists():
        raise IOError("Cannot open transform file.")

    # Read file
    with open(yaml_file, 'r') as stream:
        data_loaded = yaml.safe_load(stream)

    steps = []
    for key in data_loaded:
        # Check if the specified filter exist in tio
        if not hasattr(tio, key):
            raise AttributeError(f"The transform '{key}' is not available in tio.")
        _transfer_cls = eval('tio.' + key)

        # Get and parse the attributes
        _content = data_loaded.get(key, None)
        if _content is None:
            steps.append(_transfer_cls())
        else:
            # For list, there could be both args and kwargs.
            if isinstance(_content, list):
                _args = [i for i in _content if not isinstance(i, dict)]
                _kwargs = [i for i in _content if isinstance(i, dict)]
                _kwargs = {} if len(_kwargs) == 0 else _kwargs[0]
                steps.append(_transfer_cls(*_args, **_kwargs))

            # If its just a dict, its kwargs
            elif isinstance(_content, dict):
                steps.append(_transfer_cls(**_content))
    return tio.Compose(tuple(steps))
