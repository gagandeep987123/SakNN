'''
given Tsne reductions and original this code generates 2 stores:
1. Query Index Store
2. Result Index Store
'''
import time
from collections import defaultdict

import math
import concurrent.futures
import numpy as np


def gen_cache_gran(x_fact, y_fact):
    gran = max(x_fact, y_fact)
    if (gran < 0.5):
        return 0.5
    elif (gran < 1):
        return 1
    else:
        return math.ceil(gran)

def get_space_key(point, gran):
    #TODO
    return tuple(point//gran)

def gen_partition_data(data, embedding, gran):
    partition_space = {}
    for i in range(embedding.shape[0]):
        point_key = get_space_key(embedding[i], gran)
        if point_key not in partition_space:
            partition_space[point_key] = {}
            partition_space[point_key]["data"] = []
            partition_space[point_key]["embedding"] = []
        partition_space[point_key]["data"].append(data[i].copy().tolist())
        partition_space[point_key]["embedding"].append(embedding[i].copy().tolist())
    for key in partition_space:
        partition_space[key]["data"] = np.array(partition_space[key]["data"])
        partition_space[key]["embedding"] = np.array(partition_space[key]["embedding"])
    return partition_space

def get_nn_index(data, x, y):
    point = np.array([x, y])
    g = 0.25
    indices = np.where(np.all(data > (point - g), axis=1) & np.all(data < (point + g), axis=1))
    if len(indices[0]) != 0:
        filtered_data = data[indices]
        distances = np.sqrt(np.sum((point - filtered_data) ** 2, axis=1))
        filtered_indices = [np.argmin(distances)]  ## k =10
        return indices[0][filtered_indices]
    else:
        distances = np.sqrt(np.sum((point - data) ** 2, axis=1))
        filtered_indices = [np.argmin(distances)]
        return filtered_indices

def grid_rep_gen_task(store, i, j, part_space, part_gran):
    embd = None
    dt = None
    key = get_space_key([i, j], part_gran)
    row, col = key[0], key[1]

    for x in range(-1, 2):
        for y in range(-1, 2):
            n_row = row + x
            n_col = col + y
            key_ = tuple([n_row, n_col])
            if key_ in part_space:
                #print("check this")
                if embd is None:
                    embd = part_space[key_]["embedding"].copy()
                    dt = part_space[key_]["data"].copy()
                else:
                    embd = np.concatenate((embd, part_space[key_]["embedding"]), axis=0)
                    dt = np.concatenate((dt, part_space[key_]["data"]), axis=0)
    pt = None
    if embd is not None:
        pts = get_nn_index(embd, i, j)
        pt = dt[pts[0]].copy()
        if np.any(np.abs(embd[pts[0]] - np.array([i,j])) - part_gran >0 ):
            #out of grid
            pt = None
    return pt


def get_smart_partition_space(tsne_red, gran_result):
    partition_space = defaultdict(list)
    for i in range(tsne_red.shape[0]):
        point_key = get_space_key(tsne_red[i], 2*gran_result)
        partition_space[point_key].append(i)
    return partition_space

def get_Points(data_, indices, cord, gran):
    data = data_[indices]
    newInd = np.where(np.all(data <= cord + gran, axis=1) & np.all(data >= cord - gran, axis=1))
    return np.array(indices)[newInd[0]].tolist()

def ris_gen(store, store_enc, i, j, gran, part_space_ris, embedding, df):
    indices = []
    cord = np.array([i, j])
    key = get_space_key(cord, 2 * gran)
    row, col = key[0], key[1]
    for x in range(-1, 2):
        for y in range(-1, 2):
            n_row = row + x
            n_col = col + y
            key_ = tuple([n_row, n_col])
            if key_ in part_space_ris:
                indices.extend(part_space_ris[key_])
    pts = []
    if (len(indices) != 0):
        pts = get_Points(embedding, indices, cord, gran)
    return pts

def store_gen_task(store, store_enc, query_index_store, i, j, gran_ris, part_space_ris, part_space_qrs, gran_qrs, embedding, df):
    pts = ris_gen(store, store_enc, i, j, gran_ris, part_space_ris, embedding, df)
    grid_rep = grid_rep_gen_task(query_index_store, i, j, part_space_qrs, gran_qrs)
    if (len(pts) != 0) and grid_rep is not None:
        df_ = df[pts].copy()
        grid_ct_key = tuple([i, j])
        store[grid_ct_key] = df_
        grid_ct_key_str = str(grid_ct_key).encode()
        import encDict
        encrypted_grid_ct_key = encDict.encrypt_data_deterministic(grid_ct_key_str)
        encrypted_value = encDict.encrypt_data_nondeterministic(df_)
        store_enc[encrypted_grid_ct_key] = encrypted_value

        qrs_record = grid_rep.copy()
        qrs_record = np.append(qrs_record,i)
        qrs_record = np.append(qrs_record,j)
        query_index_store.append(qrs_record)




def stores_gen_parallel(org_data, tsne_red, div_qrs = 32,
                        gran_ris=1, workers = 25, leveled_gen=False,
                        leveldParams=None, inlevel = False):
    start_time = time.time()
    mins_emb = np.min(tsne_red, axis=0)
    maxs_emb = np.max(tsne_red, axis=0)

    x_fact = (maxs_emb[0] - mins_emb[0]) / div_qrs
    y_fact = (maxs_emb[1] - mins_emb[1]) / div_qrs

    #TODO
    gran_qrs = (maxs_emb - mins_emb) / div_qrs

    if np.any(gran_qrs > gran_ris):
        print("Some points might miss in the result set")
        return

    if inlevel:
        data_map_qrs = gen_partition_data(org_data[:,:-2], tsne_red, gran_qrs)
    else:
        data_map_qrs = gen_partition_data(org_data, tsne_red, gran_qrs)
    data_map_ris = get_smart_partition_space(tsne_red, gran_ris)
    
    result_index_store = {}
    result_index_store_enc = {}
    query_index_store = []
    errors = []

    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        # Submit tasks to the thread pool
        futures = []
        i = 0
        while i <= div_qrs:
            j = 0
            while j <= div_qrs:
                future = executor.submit(store_gen_task, result_index_store, result_index_store_enc,
                                         query_index_store, mins_emb[0] + i * x_fact, mins_emb[1] + j * y_fact,
                                         gran_ris, data_map_ris, data_map_qrs, gran_qrs, tsne_red,
                                         org_data)
                futures.append(future)
                j = j + 1
            i = i + 1

        for future in concurrent.futures.as_completed(futures):
            try:
                future.result()  # This will raise an exception if one occurred in the thread
            except Exception as exc:
                errors.append(exc)
                print(f"Task raised an exception: {exc}")

    print("All tasks completed")
    end_time = time.time()
    print("Time to create stores: ", end_time - start_time)
    if errors:
        print("\nErrors encountered during execution:")
        for error in errors:
            print(f"Error: {error}")

    leveled_stores = None
    if leveled_gen:
        start_time = time.time()
        gran_query_level1 = leveldParams[0]
        gran_result_level1 = leveldParams[1]
        print("Tsne processing")
        query_store_np = np.array(query_index_store)
        from sklearn.manifold import TSNE
        tsne = TSNE(n_components=2, random_state=42, n_jobs=25, n_iter=1500)
        X_tsne = tsne.fit_transform(query_store_np[:, :-2])
        print("Tsne processing Done")
        end_time = time.time()
        print("Tsne processing time: ", end_time - start_time)
        result_store_level1, result_store_level1_enc, query_store_level1, leveled_stores_ = (
            stores_gen_parallel(org_data=query_store_np, tsne_red=X_tsne,
                                div_qrs=gran_query_level1, gran_ris=gran_result_level1, inlevel=True)
        )
        store_lens = []
        max_size = 1
        min_size = 1000000000
        for s in result_store_level1:
            store_lens.append(result_store_level1[s].shape[0])
            max_size = max(max_size, result_store_level1[s].shape[0])
            min_size = min(min_size, result_store_level1[s].shape[0])

        print("Result Store level1 Done")
        leveled_stores = (result_store_level1, result_store_level1_enc, query_store_level1)
        end_time = time.time()
        print("Time to create leveled stores: ", end_time - start_time)

    return result_index_store, result_index_store_enc, np.array(query_index_store), leveled_stores

from performance_helper import performance_gen, performance_gen_leveled
#%%
# with open('tsne_transform/transformation_data/random_2d_scikit.npy', 'rb') as f:
#      embedding = np.load(f)
#      df = np.load(f)
# result_store, result_store_enc, query_store, leveled_stores = stores_gen_parallel(
#     org_data=df, tsne_red=embedding, div_qrs=400, gran_ris=2, leveled_gen=True, leveldParams=[100, 10]
# )
# start = time.time()
# store_ = stores_gen_parallel(org_data=df, tsne_red=embedding, gran_query=400)
# end = time.time()
# print("runtime:", (end - start))
# with open('tsne_transform/transformation_data/random_2d_scikit.npy', 'rb') as f:
#     embedding = np.load(f)
#     df = np.load(f)
#
# #query_store = query_index_gen_parallel(org_data=df, tsne_red=embedding, gran_query=200)
#
# result_store, result_store_enc, query_store,_ = stores_gen_parallel(org_data=df, tsne_red=embedding,
#                                                                       div_qrs=100, gran_ris=2)
# print("Done")
# print(len(query_store))
# print(len(result_store.keys()))
# with open('tsne_transform/transformation_data/random_2d_scikit.npy', 'rb') as f:
#     embedding = np.load(f)
#     df = np.load(f)
# result_store, result_store_enc, query_store, leveled_stores = stores_gen_parallel(org_data=df,
#                                                                                          tsne_red=embedding,
#                                                                                          div_qrs=100, gran_ris=2,
#                                                                                          leveled_gen=True,
#                                                                                          leveldParams=[100, 10])