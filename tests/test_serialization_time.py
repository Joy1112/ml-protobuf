import sys
import os.path as osp
import time
import numpy as np
from functools import wraps
sys.path.append(osp.split(osp.realpath(osp.dirname(__file__)))[0])

from ml_protobuf import array_to_proto, proto_to_array

N_RUNS = 20
MB_UNIT = 1024 * 1024


def create_array(num_mb=1024):
    desired_elements = (MB_UNIT) // 8

    shape = (num_mb, desired_elements)

    large_array = np.ones(shape, dtype=np.float64)
    memory_cost = sys.getsizeof(large_array) + large_array.nbytes
    print(f"The array takes about {memory_cost / (1024 * MB_UNIT):.3f} GB.")

    return large_array


def split_array(array, batch_size):
    for start in range(0, array.shape[0], batch_size):
        end = start + batch_size if start + batch_size < array.shape[0] else array.shape[0]
        yield array[start:end]

def time_counter_decorator(n_runs=N_RUNS):
    def decorator(fn):
        def wrapper(*args, **kwargs):
            total_time_cost = 0.
            for i in range(n_runs):
                start_time = time.perf_counter()
                returns = fn(*args, **kwargs)
                end_time = time.perf_counter()
                total_time_cost += end_time - start_time

            avg_time_cost = total_time_cost / n_runs * 1000
            print(f"time spent in func `{fn.__name__}()`: {avg_time_cost:.5f} ms.")

            return returns, avg_time_cost
        
        return wrapper
    
    return decorator


# --------------------------- serialization funcs ---------------------------
@time_counter_decorator()
def serialize_array_ori(array):
    proto = array_to_proto(array)

    return proto


@time_counter_decorator()
def serialize_array_split(array, batch_size):
    proto_list = [
        array_to_proto(sub_array) for sub_array in split_array(array, batch_size=batch_size)
    ]
    
    return proto_list


@time_counter_decorator()
def serialize_array_slice(array):
    proto_list = [
        array_to_proto(array[idx:idx + 1]) for idx in range(array.shape[0])
    ]
    
    return proto_list


@time_counter_decorator()
def serialize_array_index(array):
    proto_list = [
        array_to_proto(array[idx]) for idx in range(array.shape[0])
    ]
    return proto_list


# --------------------------- deserialization funcs ---------------------------
@time_counter_decorator()
def deserialize_array_ori(proto):
    return proto_to_array(proto)


@time_counter_decorator()
def deserialize_array_list_concat(proto_list):
    return np.concatenate([
        proto_to_array(proto) for proto in proto_list
    ])


@time_counter_decorator()
def deserialize_array_list_stack(proto_list):
    return np.vstack([
        proto_to_array(proto) for proto in proto_list
    ])


def test_time(num_mb, serialize_fn, deserialize_fn, **kwargs):
    array = create_array(num_mb=num_mb)

    start_time = time.perf_counter()
    proto_var, time_cost_se = serialize_fn(array, **kwargs)
    de_array, time_cost_de = deserialize_fn(proto_var)
    end_time = time.perf_counter()

    assert np.array_equal(array, de_array)
    
    time_cost = 1000 * (end_time - start_time) / N_RUNS
    print(f"total time spent for {num_mb}MB array with kwargs={kwargs}: {time_cost:.5f} ss.")

    del array, proto_var, de_array

    return np.array([time_cost_se, time_cost_de, time_cost])


if __name__ == "__main__":
    test_official_serializate_funcs = True
    if test_official_serializate_funcs:
        def test_array_singlethread(num_mb):
            test_time(num_mb, serialize_array_ori, deserialize_array_ori)
            test_time(num_mb, serialize_array_split, deserialize_array_list_concat, batch_size=1024)
            test_time(num_mb, serialize_array_split, deserialize_array_list_concat, batch_size=4)
            # test_time(num_mb, serialize_array_split, deserialize_array_list_concat, batch_size=1)
            test_time(num_mb, serialize_array_slice, deserialize_array_list_concat)
            test_time(num_mb, serialize_array_index, deserialize_array_list_stack)
        test_array_singlethread(num_mb=10)
        # test_array_singlethread(num_mb=102)
        # test_array_singlethread(num_mb=1024)
        # test_array_singlethread(num_mb=1024*5)


