import sys
import os
import os.path as osp
import time
import math
import pickle
import pandas as pd
import numpy as np
from functools import wraps
sys.path.append(osp.split(osp.realpath(osp.dirname(__file__)))[0])

from ml_protobuf import array_to_proto, proto_to_array

N_RUNS = 20
MB_UNIT = 1024 * 1024
# SEDE_TOOL = 'pickle'
SEDE_TOOL = 'numpy'


def create_array(num_mb=1024):
    desired_elements = (MB_UNIT) // 8

    shape = (num_mb, desired_elements)

    large_array = np.ones(shape, dtype=np.float64)
    # memory_cost = sys.getsizeof(large_array) + large_array.nbytes
    print(f"The array takes about {large_array.nbytes / (1024 * MB_UNIT):.3f} GB Memory.")

    return large_array


def split_array(array, batch_size):
    for start in range(0, array.shape[0], batch_size):
        end = start + batch_size if start + batch_size < array.shape[0] else array.shape[0]
        yield array[start:end]


def time_counter(n_runs=N_RUNS):
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

        wrapper.__name__ = str(fn.__name__)
        
        return wrapper
    
    return decorator


# --------------------------- serialization funcs ---------------------------
@time_counter()
def serialize_array_ori(array):
    proto = array_to_proto(array, dump_tool=SEDE_TOOL)

    return proto


@time_counter()
def serialize_array_split(array, batch_size):
    proto_list = [
        array_to_proto(sub_array, dump_tool=SEDE_TOOL) for sub_array in split_array(array, batch_size=batch_size)
    ]
    
    return proto_list


@time_counter()
def serialize_array_slice(array):
    proto_list = [
        array_to_proto(array[idx:idx + 1], dump_tool=SEDE_TOOL) for idx in range(array.shape[0])
    ]
    
    return proto_list


@time_counter()
def serialize_array_index(array):
    proto_list = [
        array_to_proto(array[idx], dump_tool=SEDE_TOOL) for idx in range(array.shape[0])
    ]
    return proto_list


# --------------------------- deserialization funcs ---------------------------
@time_counter()
def deserialize_array_ori(proto):
    return proto_to_array(proto, load_tool=SEDE_TOOL)


@time_counter()
def deserialize_array_list_concat(proto_list):
    return np.concatenate([
        proto_to_array(proto, load_tool=SEDE_TOOL) for proto in proto_list
    ])


@time_counter()
def deserialize_array_list_stack(proto_list):
    return np.vstack([
        proto_to_array(proto, load_tool=SEDE_TOOL) for proto in proto_list
    ])


def test_time(num_mb, serialize_fn, deserialize_fn, **kwargs):
    array = create_array(num_mb=num_mb)

    start_time = time.perf_counter()
    proto_var, time_cost_se = serialize_fn(array, **kwargs)
    de_array, time_cost_de = deserialize_fn(proto_var)
    end_time = time.perf_counter()

    assert np.array_equal(array, de_array)
    
    time_cost = 1000 * (end_time - start_time) / N_RUNS
    print(f"total time spent for {num_mb}MB array by `{serialize_fn.__name__}(), {deserialize_fn.__name__}()` with kwargs={kwargs}: {time_cost:.5f} ms.")

    del array, proto_var, de_array

    return np.array([time_cost_se, time_cost_de, time_cost])


if __name__ == "__main__":
    categories = [
        'direct',
        'split_1G',
        'split_4M',
        'split_1M',
        'slice_1M',
        'index_1M'
    ]
    time_cost_dict = {k: [] for k in categories}


    def test_array_singlethread(num_mb):
        time_cost_dict['direct'].append(test_time(num_mb, serialize_array_ori, deserialize_array_ori))
        time_cost_dict['split_1G'].append(test_time(num_mb, serialize_array_split, deserialize_array_list_concat, batch_size=1024))
        time_cost_dict['split_4M'].append(test_time(num_mb, serialize_array_split, deserialize_array_list_concat, batch_size=4))
        time_cost_dict['split_1M'].append(test_time(num_mb, serialize_array_split, deserialize_array_list_concat, batch_size=1))
        time_cost_dict['slice_1M'].append(test_time(num_mb, serialize_array_slice, deserialize_array_list_concat))
        time_cost_dict['index_1M'].append(test_time(num_mb, serialize_array_index, deserialize_array_list_stack))


    array_sizes = np.array([10, 102, 1024, 1024 * 5, 1024 * 10])
    for num_mb in array_sizes:
        test_array_singlethread(num_mb)
    
    for k in categories:
        time_cost_dict[k] = np.vstack(time_cost_dict[k])


    def plot_results(df, fig_name):
        import seaborn as sns
        import matplotlib.pyplot as plt
        
        sns.set(style='whitegrid')
        sns.set_palette("Set2")
    
        plt.figure(figsize=(10, 6))
        plt.xlabel('Array Size (GB): $10^x$')
        plt.ylabel('Time Cost (ms)')
        sns.lineplot(data=df, marker='o', dashes=False)
        plt.title(fig_name)
        plt.legend(title='Methods')

        plt.savefig(osp.join(save_path, f'{fig_name} with numpy tobytes().jpg'))


    fig_name = ['serialization time', 'deserialization time', 'total time cost']

    save_path = osp.join(osp.realpath(osp.dirname(__file__)), 'results')
    if not osp.exists(save_path):
        os.mkdir(save_path)

    log_array_sizes = np.log10(array_sizes / 1024.)
    for i in range(len(fig_name)):
        df = pd.DataFrame(data={k: time_cost_dict[k][:, i].reshape(-1) for k in categories}, index=log_array_sizes)
        df.to_csv(osp.join(save_path, fig_name[i] + '.csv'))

        plot_results(df, fig_name[i])
