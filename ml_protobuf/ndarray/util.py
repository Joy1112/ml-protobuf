# Copyright 2023 Joy1112
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
import pickle

from typing import Tuple, Optional

from . import array_pb2


def array_to_proto(array: np.ndarray, shape: Optional[Tuple[int]] = None, dtype: Optional[str] = None, dump_tool: str = 'numpy') -> array_pb2.NDArray:
    """
    Serializes a numpy array into a NDArray protobuf message.

    Args:
        array: np.ndarray
            The numpy array to serialize.
        shape: tuple, optional
            The shape of the numpy array to convert to, defaults to None.
        dtype: str, optional
            The dtype of the numpy array to convert to, defaults to None.
            Refer to 'src/grpc/protos/array.proto': DType.

    Returns:
        array_pb2.NDArray: The NDArray Proto containing the given array.

    Note:
        a) python strings should be utf-8 and the bytes will automatically be converted to utf-8.
        b) dtype == 'string' & 'object' will be treated the same. dtype = 'object' is recommanded (can handle elements of NoneType),
           but it only supported the type of elements to be None, str or bytes. Complex object is not supported yet.
    """
    assert isinstance(array, np.ndarray), f"array must be of type np.ndarray, but got {type(array)}."

    # shape
    if shape is None:
        shape = array.shape

    # dtype
    if dtype is None:
        dtype = str(array.dtype)
    else:
        dtype = dtype.lower()
        array = array.astype(np.dtype(dtype))

    if dtype.upper() not in array_pb2.DType.keys():
        raise TypeError(f"Only {array_pb2.DType.keys()} data types are supported, but got {dtype.upper()}.")

    dtype_dict = {k.lower(): v for k, v in array_pb2.DType.items()}

    # create array proto
    array_proto = array_pb2.NDArray()

    array_proto.shape.extend(list(shape))
    array_proto.dtype = dtype_dict[dtype]

    if dtype not in ['string', 'object']:
        # serialize the array data
        if dump_tool == 'numpy':
            array_proto.data = array.tobytes()
        elif dump_tool == 'pickle':
            array_proto.data = pickle.dumps(array.tostring(), protocol=pickle.HIGHEST_PROTOCOL)
        else:
            raise ValueError
    else:
        # dtype == 'object' or dtype == 'string' are similar in this projects
        # e.g.: Batch.done - np.array([None, 'bad_transition'], dtype=np.object)
        # The type of the elements in the array must be None or str or bytes, complex objects are not supported yet.
        str_list = []
        none_list = []
        for value in array.flatten():
            if value is None:
                str_list.append('')
                none_list.append(True)
            elif isinstance(value, str):
                str_list.append(value)
                none_list.append(False)
            elif isinstance(value, bytes):
                str_list.append(value.decode('utf-8'))
                none_list.append(False)
            else:
                raise TypeError(f"object elements in array must be of type str or bytes but got {type(value)}")
        array_proto.string_data.extend(str_list)
        array_proto.none_object.extend(none_list)

    return array_proto


def proto_to_array(array_proto: array_pb2.NDArray, load_tool: str = 'numpy') -> np.ndarray:
    """
    Retrieve the np.ndarray from the NDArray Proto data.

    Args:
        array_proto: array_pb2.NDArray
            The NDArray Proto data.

    Returns:
        np.ndarray: The np.ndarray retrieved from NDArray Proto.
    """
    assert isinstance(array_proto, array_pb2.NDArray), f"The array_proto must be a NDArray Proto data, but got {type(array_proto)}."

    shape = [dim for dim in array_proto.shape]

    dtype_reversed_dict = {v: k.lower() for k, v in array_pb2.DType.items()}
    dtype = dtype_reversed_dict[int(array_proto.dtype)]

    if dtype not in ['string', 'object']:
        if load_tool == 'numpy':
            array = np.frombuffer(array_proto.data, dtype=np.dtype(dtype)).reshape(shape)
        elif load_tool == 'pickle':
            array = np.frombuffer(pickle.loads(array_proto.data), dtype=np.dtype(dtype)).reshape(shape)
        else:
            raise ValueError
    else:
        # the string will be decoded as utf-8.
        array = np.array(array_proto.string_data, dtype=object)
        for i in range(len(array_proto.none_object)):
            if array_proto.none_object[i]:
                array[i] = None
        array = array.reshape(shape)

    return array
