import ml_protobuf

import numpy as np
import torch

import time


N_RUNS = 20


def getArray(num_mb=1024):
    desired_elements = (1024 * 1024) // 8

    shape = (num_mb, desired_elements)

    large_array = np.ones(shape, dtype=np.float64)
    print(f"The array takes about {large_array.nbytes/(1024*1024*1024):.2f} GB")

    return large_array


import services
from services.data_type.array import arrayToProto, protoToArray
from services.data_type.torch_tensor import tensorToProto, protoToTensor
import utils


def split_array(array, batch_size):
    for start in range(0, array.shape[0], batch_size):
        end = start + batch_size if start + batch_size < array.shape[0] else array.shape[0]
        yield array[start:end]

@utils.time_counter_decorator(n_runs=N_RUNS)
def serialize_array_ori(array, **kwargs):
    proto = arrayToProto(array, **kwargs)

    return proto
    

@utils.time_counter_decorator(n_runs=N_RUNS)
def serialize_array_split(array, batch_size):
    proto_list = [
        arrayToProto(sub_array) for sub_array in split_array(array, batch_size=batch_size)
    ]
    
    return proto_list

@utils.time_counter_decorator(n_runs=N_RUNS)
def serialize_array_slice(array):
    proto_list = [
        arrayToProto(array[idx:idx + 1]) for idx in range(array.shape[0])
    ]
    
    return proto_list


@utils.time_counter_decorator(n_runs=N_RUNS)
def serialize_array_index(array):
    proto_list = [
        arrayToProto(array[idx]) for idx in range(array.shape[0])
    ]
    return proto_list
    

@utils.time_counter_decorator(n_runs=N_RUNS)
def deserialize_array_ori(proto, **kwargs):
    return protoToArray(proto, **kwargs)


@utils.time_counter_decorator(n_runs=N_RUNS)
def deserialize_array_list(proto_list):
    array_list = []
    for proto in proto_list:
        array_list.append(protoToArray(proto))

    if len(array_list[0].shape) == 1:
        return np.vstack(array_list)
    else:
        return np.concatenate(array_list)






# from multiprocessing import reduction
# import cloudpickle
# reduction.ForkingPickler = cloudpickle.Pickler


# import multiprocessing as mpi
# # mpi.set_start_method('fork', True)
# # import multiprocess as mpi
# from services.data_type.array import array_pb2
# from services.data_type.array.array_pb2 import NDArray


# def serialize_array_mpi_queue(array_list, result_queue):
#     for array in array_list:
#         proto = arrayToProto(array)

#         result_queue.put(proto)


# def serialize_array_mpi_pipe(data_queue, child_conn=None):
#     proto = arrayToProto(data_queue.get())

#     child_conn.send(proto)


# @utils.time_counter_decorator(n_runs=N_RUNS)
# def multi_process_serialize_array_mpi_queue(array, batch_size=1, pool_size=5):
#     result_queue = mpi.Queue()

#     array_list = [sub_array for sub_array in split_array(array, batch_size=batch_size)]
#     partition_size = len(array_list) // pool_size

#     pool = []
#     for i in range(pool_size):
#         start = i * partition_size
#         end = (i + 1) * partition_size if (i < pool_size - 1) else len(array_list)
#         p = mpi.Process(target=serialize_array_mpi_queue, args=(array_list[start:end], result_queue, ))
#         p.start()
#         pool.append(p)

#     for p in pool:
#         p.join()

#     proto_list = []
#     while not result_queue.empty():
#         print("queue is not empty")
#         r = result_queue.get()
#         print(r)
#         proto_list.append(r)

#     return proto_list


# @utils.time_counter_decorator(n_runs=N_RUNS)
# def multi_process_serialize_array_mpi_pipe_oneside(array, batch_size=1, pool_size=5):
#     # parrent_conn, child_conn = mpi.Pipe(duplex=False)
#     parrent_conn, child_conn = mpi.Pipe()

#     array_list = [sub_array for sub_array in split_array(array, batch_size=batch_size)]

#     pool = [
#         mpi.Process(target=serialize_array_mpi, args=(sub_array, child_conn, 'pipe')) for sub_array in array_list
#     ]

#     for p in pool:
#         p.daemon = True
#         p.start()

#     for p in pool:
#         p.join()

#     count = 0
#     proto_list = []
#     while count < len(array_list):
#         try:
#             msg = parrent_conn.recv()
#             proto_list.append(msg)
#             count += 1
#         except EOFError:
#             break

#     return proto_list


def test_time(num_mb, serialize_fn, deserialize_fn, **kwargs):
    
    array = getArray(num_mb=num_mb)

    start_time = time.perf_counter()
    proto_var = serialize_fn(array, **kwargs)
    de_array = deserialize_fn(proto_var)
    end_time = time.perf_counter()

    assert np.array_equal(array, de_array)

    print(f"total time spent for {num_mb}MB array with kwargs={kwargs}: \t\t\t", (end_time-start_time)/N_RUNS)
    del array, proto_var, de_array


if __name__ == "__main__":
    test_official_serializate_funcs = True
    if test_official_serializate_funcs:
        def test_array_singlethread(num_mb):
            test_time(num_mb, serialize_array_ori, deserialize_array_ori)
            test_time(num_mb, serialize_array_split, deserialize_array_list, batch_size=1024)
            test_time(num_mb, serialize_array_split, deserialize_array_list, batch_size=4)
            # test_time(num_mb, serialize_array_split, deserialize_array_list, batch_size=1)
            test_time(num_mb, serialize_array_slice, deserialize_array_list)
            test_time(num_mb, serialize_array_index, deserialize_array_list)
        test_array_singlethread(num_mb=10)
        test_array_singlethread(num_mb=102)
        test_array_singlethread(num_mb=1024)
        test_array_singlethread(num_mb=1024*5)

    # test_mpi_funcs = False
    # if test_mpi_funcs:
    #     test_time(1024, multi_process_serialize_array_mpi_queue, deserialize_array_list, batch_size=10, pool_size=5)
        # test_time(1024, multi_process_serialize_array_mpi_pipe_oneside, deserialize_array_list, batch_size=1024, pool_size=5)

