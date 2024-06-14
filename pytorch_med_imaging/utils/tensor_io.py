import torch
import numpy as np
import base64
import json
from typing import Optional, Union

def tensor_to_ascii(tensor: torch.Tensor, to_dict: Optional[bool] = False) -> Union[str, dict]:
    r"""Converts a PyTorch tensor into a JSON-formatted ASCII string.

    This function first transfers the tensor to CPU memory, if not already in CPU. It then converts the tensor
    to a NumPy array and serializes this array to bytes. The bytes are encoded into a Base64 ASCII string,
    which is then packaged into a JSON string along with metadata about the tensor's data type and shape.

    Args:
        tensor (torch.Tensor): The tensor to be converted.

    Returns:
        str: A JSON string containing the Base64-encoded data of the tensor, its data type, and its shape.
             The JSON string has keys 'data', 'dtype', and 'shape'.

    Example:
        >>> tensor = torch.tensor([1, 2, 3])
        >>> ascii_representation = tensor_to_ascii(tensor)
        >>> print(ascii_representation)
        {"data": "AQAAAAIAAAADAAAA", "dtype": "int64", "shape": [3]}
    """
    # Convert the tensor to a NumPy array
    numpy_array = tensor.cpu().numpy()
    # Serialize the NumPy array to bytes
    byte_data = numpy_array.tobytes()
    # Encode the bytes to a Base64 string
    base64_bytes = base64.b64encode(byte_data)
    # Convert the Base64 bytes to an ASCII string
    ascii_string = base64_bytes.decode('ascii')
    # Create a dictionary to store the ASCII string, dtype, and shape
    tensor_info = {
        'data': ascii_string,
        'dtype': str(numpy_array.dtype),
        'shape': numpy_array.shape
    }
    if not to_dict:
        # Serialize the dictionary to a JSON string
        tensor_json = json.dumps(tensor_info)
        return tensor_json
    else:
        return tensor_info


def ascii_to_tensor(tensor_json: str) -> torch.Tensor:
    r"""Converts a JSON-formatted ASCII string back into a PyTorch tensor.

    This function deserializes a JSON string that contains a Base64-encoded ASCII string representation of tensor data,
    along with metadata about the tensor's data type and shape. The function decodes this information to reconstruct
    the original tensor.

    Args:
        tensor_json (str): A JSON string containing the Base64-encoded data of the tensor, its data type, and its shape.
                           The JSON string should have keys 'data', 'dtype', and 'shape'.

    Returns:
        torch.Tensor: The PyTorch tensor reconstructed from the provided ASCII-encoded JSON string.

    Example:
        >>> tensor_json = '{"data": "AQAAAAIAAAADAAAA", "dtype": "int64", "shape": [3]}'
        >>> tensor = ascii_to_tensor(tensor_json)
        >>> print(tensor)
        tensor([1., 2., 3.])

    Raises:
        KeyError: If the expected keys ('data', 'dtype', 'shape') are missing from the JSON string.
        ValueError: If decoding the data fails or if the JSON string cannot be parsed.
    """
    # Deserialize the JSON string to a dictionary
    tensor_info = json.loads(tensor_json)
    # Extract the ASCII string, dtype, and shape from the dictionary
    ascii_string = tensor_info['data']
    dtype = np.dtype(tensor_info['dtype'])
    shape = tuple(tensor_info['shape'])
    # Convert the ASCII string to Base64 bytes
    base64_bytes = ascii_string.encode('ascii')
    # Decode the Base64 bytes to raw bytes
    byte_data = base64.b64decode(base64_bytes)
    # Convert the bytes back to a NumPy array
    numpy_array = np.frombuffer(byte_data, dtype=dtype)
    # Reshape the NumPy array
    numpy_array = numpy_array.reshape(shape)
    # Convert the NumPy array back to a PyTorch tensor
    tensor = torch.tensor(numpy_array, dtype=torch.float32)
    return tensor