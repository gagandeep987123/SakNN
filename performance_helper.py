import pickle

import numpy as np
import concurrent.futures

import encDict, time
from pympler import asizeof


def get_min_store_cord(store, pt):
    y = np.sqrt(np.sum((store[:,:-2]-pt)**2,axis=1))
    minimum = np.argmin(y)
    return store[minimum,-2:]

def getKNNIndices(data, point,k=1000, isEnc=False):
    if  isEnc:
        data = encDict.decrypt(data)
    distances = np.sqrt(np.sum((point-data)**2,axis=1))
    indices = np.argsort(distances)[:k] ## k =10
    return data[indices]


def OAR(sub_data, org_data, point, k):
    k= min(sub_data.shape[0], k)
    if k == 0:
        return 1
    distances_sub = np.sqrt(np.sum((sub_data - point) ** 2, axis=1))
    distances_sub = distances_sub + 0.0000000000001
    distances_org = np.sqrt(np.sum((org_data - point) ** 2, axis=1))
    distances_org = distances_org + 0.0000000000001
    distances_sub.sort()
    distances_org.sort()
    distances_sub = distances_sub[1:k]
    distances_org = distances_org[1:k]
    ratios = distances_sub / distances_org
    ratios5, ratios10, ratios20, ratios30, ratios40, ratios50 = (np.average(ratios[1:6]),
                                                                 np.average(ratios[1:11]), np.average(ratios[1:21]),
                                                                 np.average(ratios[1:31]), np.average(ratios[1:41]),
                                                                 np.average(ratios[1:51]))
    return ratios5, ratios10, ratios20, ratios30, ratios40, ratios50

def performance_task(recalls50, oars50, results, iteration, org_data, point, query_store, result_store, result_store_enc,
                     recalls5, recalls10, recalls20, recalls30, recalls40,
                     oars5, oars10, oars20, oars30, oars40, query_sizes, result_sizes, isEnc, timeCalFlag):
    grid_rep = get_min_store_cord(query_store, point)
    true_50NN = None
    if not timeCalFlag:
        true_50NN = getKNNIndices(org_data, point, k=51)

    if not isEnc:
        from_stor50NN = getKNNIndices(
            result_store[tuple(grid_rep)],
            #result_store_enc[grid_rep_enc],
            point,
            k=51
        )
    else:
        grid_rep_enc = encDict.encrypt_data_deterministic(str(tuple(grid_rep)).encode())
        if not timeCalFlag:
            query_sizes[iteration] = asizeof.asizeof(grid_rep_enc)
            result_sizes[iteration] = asizeof.asizeof(result_store_enc[grid_rep_enc])
        from_stor50NN = getKNNIndices(
            # result_store[tuple(grid_rep)],
            result_store_enc[grid_rep_enc],
            point,
            k=51,
            isEnc=True
        )

    if from_stor50NN.shape[0] >= 51: ##k
        results[iteration] = True

    if not timeCalFlag:
        ans_set = set()
        flag_50 = np.zeros(51)

        for knn_data in from_stor50NN:
            ans_set.add(tuple(knn_data))

        for i, knn_data in enumerate(true_50NN):
            if tuple(knn_data) in ans_set:
                flag_50[i] = 1
        oars5[iteration],oars10[iteration],oars20[iteration],oars30[iteration],oars40[iteration],oars50[iteration] = (
            OAR(result_store[tuple(grid_rep)], org_data, point, k=51))
        #flag_50 = np.array(flag_50)
        recalls5[iteration] = np.sum(flag_50[1:6])
        recalls10[iteration] = np.sum(flag_50[1:11])
        recalls20[iteration] = np.sum(flag_50[1:21])
        recalls30[iteration] = np.sum(flag_50[1:31])
        recalls40[iteration] = np.sum(flag_50[1:41])
        recalls50[iteration] = np.sum(flag_50[1:51])




def performance_gen(org_data, query_store, result_store, result_store_enc, iterations = 1000, isEnc = False, timeCalFlag = False):
    recalls50 = [0] * iterations
    oars50 = [0] * iterations
    query_sizes = [0]*iterations
    result_sizes = [0]*iterations

    recalls5 = [0] * iterations
    oars5 = [0] * iterations

    recalls10 = [0] * iterations
    oars10 = [0] * iterations

    recalls20 = [0] * iterations
    oars20 = [0] * iterations

    recalls30 = [0] * iterations
    oars30 = [0] * iterations

    recalls40 = [0] * iterations
    oars40 = [0] * iterations

    results = [False]*iterations
    indices = np.random.permutation(list(range(org_data.shape[0])))
    futures = []
    if not timeCalFlag:
        with concurrent.futures.ThreadPoolExecutor(max_workers=25) as executor:
            for iteration in range(iterations):
                point = org_data[indices[iteration]]
                future = executor.submit(performance_task, recalls50, oars50, results, iteration, org_data, point,
                                         query_store, result_store, result_store_enc,
                                         recalls5, recalls10, recalls20, recalls30, recalls40,
                                         oars5, oars10, oars20, oars30, oars40, query_sizes, result_sizes,
                                         isEnc, timeCalFlag)
                futures.append(future)
        print("All Tasks done")
        return oars50, recalls50, results, recalls5, recalls10, recalls20, recalls30, recalls40, oars5, oars10, oars20, oars30, oars40, result_sizes, query_sizes
    else:
        start_time = time.time()
        for iteration in range(iterations):
            point = org_data[indices[iteration]]
            performance_task(recalls50, oars50, results, iteration, org_data, point,
                             query_store, result_store, result_store_enc,
                             recalls5, recalls10, recalls20, recalls30, recalls40,
                             oars5, oars10, oars20, oars30, oars40, query_sizes, result_sizes,
                             isEnc, timeCalFlag)
        end_time = time.time()
        print("Time taken:", end_time - start_time)
        print("All Tasks done")
        return oars50, recalls50, results, recalls5, recalls10, recalls20, recalls30, recalls40, oars5, oars10, oars20, oars30, oars40 ,result_sizes, query_sizes

def getKNNIndices_(data, point,k=1000, isEnc = False):
    if isEnc:
        data = encDict.decrypt(data)
    distances = np.sqrt(np.sum((point-data)**2,axis=1))
    indices = np.argsort(distances)[:k] ## k =10
    return indices

def performance_level_task(recalls50, oars50, results, iteration, org_data, point, result_store,
                           result_store_enc,
                           result_store_level1, result_store_level1_enc, query_store_level1,
                           recalls5, recalls10, recalls20, recalls30, recalls40,
                           oars5, oars10, oars20, oars30, oars40,
                           query_sizes1, query_sizes2,result_sizes1,result_sizes2,
                           isEnc, timeCalFlag):
    grid_rep_level1 = get_min_store_cord(query_store_level1, point)
    if isEnc:
        grid_rep_enc = encDict.encrypt_data_deterministic(str(tuple(grid_rep_level1)).encode())
        #print("Size of query grid_rep_level1:", grid_rep_enc.nbytes)
        data_next_step = result_store_level1_enc[grid_rep_enc]
        if not timeCalFlag:
            query_sizes1[iteration] = asizeof.asizeof(grid_rep_enc)
            result_sizes1[iteration] = asizeof.asizeof(data_next_step)
        data_next_step = encDict.decrypt(data_next_step)
    else:
        data_next_step = result_store_level1[tuple(grid_rep_level1)]
    grid_rep_index = getKNNIndices_(data_next_step[:, :-2], point, k=1)
    grid_rep = result_store_level1[tuple(grid_rep_level1)][grid_rep_index, -2:][0]

    #grid_rep = getMinStoreCord(np.array(query_store), point)
    if not timeCalFlag:
        true_50NN = getKNNIndices(org_data, point, k=51)
        data_next_step = result_store[tuple(grid_rep)]
    if isEnc:
        enc_key = encDict.encrypt_data_deterministic(str(tuple(grid_rep)).encode())
        data_next_step = result_store_enc[enc_key]
        if not timeCalFlag:
            query_sizes2[iteration] = asizeof.asizeof(enc_key)
            result_sizes2[iteration] = asizeof.asizeof(data_next_step)
    from_stor50NN = getKNNIndices(data_next_step,point, k=51,isEnc=isEnc)
    if not timeCalFlag:
        ans_set = set()
        flag_50 = np.zeros(51)

        if from_stor50NN.shape[0] >= 51:  ##k
            results[iteration] = True

        for knn_data in from_stor50NN:
            ans_set.add(tuple(knn_data))

        for i, knn_data in enumerate(true_50NN):
            if tuple(knn_data) in ans_set:
                flag_50[i] = 1
        oars5[iteration], oars10[iteration], oars20[iteration], oars30[iteration], oars40[iteration], oars50[iteration] = (
            OAR(result_store[tuple(grid_rep)], org_data, point, k=51))
        #flag_50 = np.array(flag_50)
        recalls5[iteration] = np.sum(flag_50[1:6])
        recalls10[iteration] = np.sum(flag_50[1:11])
        recalls20[iteration] = np.sum(flag_50[1:21])
        recalls30[iteration] = np.sum(flag_50[1:31])
        recalls40[iteration] = np.sum(flag_50[1:41])
        recalls50[iteration] = np.sum(flag_50[1:51])
    # for knn_data in true_50NN:
    #     org_set.add(tuple(knn_data))
    # i =0
    # if from_stor50NN.shape[0] >= 50: ##k
    #     results[iteration] = True
    # for knn_data in from_stor50NN:
    #     if tuple(knn_data) in org_set:
    #         i += 1
    # oars50[iteration] = OAR(result_store[tuple(grid_rep)], org_data, point, k=50)
    # recalls50[iteration] = i

def performance_gen_leveled(org_data, result_store, result_store_enc,
                            query_store_level1,result_store_level1,result_store_level1_enc,
                            iterations = 1000,
                            isEnc=False,
                            timeCalFlag=False):
    query_sizes1 = [0]  * iterations
    query_sizes2 = [0] * iterations
    result_sizes1 = [0] * iterations
    result_sizes2 = [0] * iterations

    recalls50 = [0] * iterations
    oars50 = [0] * iterations

    recalls5 = [0] * iterations
    oars5 = [0] * iterations

    recalls10 = [0] * iterations
    oars10 = [0] * iterations

    recalls20 = [0] * iterations
    oars20 = [0] * iterations

    recalls30 = [0] * iterations
    oars30 = [0] * iterations

    recalls40 = [0] * iterations
    oars40 = [0] * iterations

    # print("Tsne processing")
    # query_store_np = np.array(query_store)
    # from sklearn.manifold import TSNE
    # tsne = TSNE(n_components=2, random_state=42, n_jobs=25, n_iter=1500)
    # X_tsne = tsne.fit_transform(query_store_np[:, :-2])
    # print("Tsne processing Done")
    # print("Result Store level1")
    # import stores
    # result_store_level1, result_store_level1_enc = stores.stores_gen_parallel(org_data=query_store_np, tsne_red=X_tsne,
    #                                                                           div_qrs=gran_query_level1, gran_ris=gran_result_level1)
    # #print("number of bytes by result store level1:", result_store_level1_enc.nbytes)
    # print("Storage size of result store level1:", asizeof.asizeof(result_store_level1_enc))
    # store_lens = []
    # max_size = 1
    # min_size = 1000000000
    # for s in result_store_level1:
    #     store_lens.append(result_store_level1[s].shape[0])
    #     max_size = max(max_size, result_store_level1[s].shape[0])
    #     min_size = min(min_size, result_store_level1[s].shape[0])
    # print(sum(store_lens))
    # print(len(store_lens))
    # print(sum(store_lens) / len(store_lens))
    # print(max_size)
    # print(min_size)
    # print("Result Store level1 Done")
    # print("query Store level1")
    # query_store_level1 = stores.query_index_gen_parallel(org_data=query_store_np[:, :-2], tsne_red=X_tsne,
    #                                                     gran_query=gran_query_level1)


    results = [False]*iterations
    indices = np.random.permutation(list(range(org_data.shape[0])))
    futures = []
    if not timeCalFlag:
        with concurrent.futures.ThreadPoolExecutor(max_workers=25) as executor:
            for iteration in range(iterations):
                point = org_data[indices[iteration]]
                future = executor.submit(
                    performance_level_task,
                    recalls50, oars50, results, iteration, org_data, point,
                    result_store, result_store_enc, result_store_level1,
                    result_store_level1_enc,
                    query_store_level1,
                    recalls5, recalls10, recalls20, recalls30, recalls40,
                    oars5, oars10, oars20, oars30, oars40,
                    query_sizes1, query_sizes2,result_sizes1,result_sizes2,
                    isEnc, timeCalFlag
                )
                futures.append(future)
        print("All Tasks done")
        return (oars50, recalls50, results, recalls5, recalls10, recalls20, recalls30, recalls40, oars5,
                oars10, oars20, oars30, oars40,
                query_sizes1, query_sizes2, result_sizes1, result_sizes2)
    else:
        start_time = time.time()
        for iteration in range(iterations):
            point = org_data[indices[iteration]]
            performance_level_task(
                recalls50, oars50, results, iteration, org_data, point,
                result_store, result_store_enc,result_store_level1,
                    result_store_level1_enc,
                    query_store_level1,
                recalls5, recalls10, recalls20, recalls30, recalls40,
                oars5, oars10, oars20, oars30, oars40,
                query_sizes1, query_sizes2, result_sizes1, result_sizes2,
                isEnc, timeCalFlag
            )
        end_time = time.time()
        print("Time taken:", end_time - start_time)
        print("All Tasks done")
        return (oars50, recalls50, results, recalls5, recalls10, recalls20, recalls30, recalls40, oars5,
                oars10, oars20, oars30, oars40,
                query_sizes1, query_sizes2, result_sizes1, result_sizes2)


# import stores, os
# directory = "cache"
# os.makedirs(directory, exist_ok=True)
# with open('tsne_transform/transformation_data/random_2d_scikit.npy', 'rb') as f:
#     embedding = np.load(f)
#     df = np.load(f)
# ris_path = os.path.join(directory, 'rs_store')
# qrs_path = os.path.join(directory, 'qr_store')
# ris_path_enc = os.path.join(directory, 'rs_store_enc')
#
# qrs_level_path = os.path.join(directory, 'qr_level_store')
# ris_level_path = os.path.join(directory, 'rs_level_store')
# ris_level_path_enc = os.path.join(directory, 'rs_level_store_enc')
#
# if os.path.exists("cache/rs_store.npy"):
#     result_store = np.load(ris_path + '.npy', allow_pickle=True).item()
#     result_store_enc = np.load(ris_path_enc + '.npy', allow_pickle=True).item()
#     query_store = np.load(qrs_path + '.npy', allow_pickle=True)
#     query_store_level1 = np.load(qrs_level_path + '.npy', allow_pickle=True)
#     result_store_level1 = np.load(ris_level_path + '.npy', allow_pickle=True).item()
#     result_store_level1_enc = np.load(ris_level_path_enc + '.npy', allow_pickle=True).item()
#     leveled_stores = [result_store_level1, result_store_level1_enc,query_store_level1]
# else:
#     result_store, result_store_enc, query_store, leveled_stores = (
#         stores.stores_gen_parallel(
#             org_data=df, tsne_red=embedding,
#             div_qrs=100, gran_ris=2, leveled_gen=True,
#             leveldParams=[100,10]
#         )
#     )
#     np.save(ris_path, result_store, allow_pickle=True)
#     np.save(ris_path_enc, result_store_enc, allow_pickle=True)
#     np.save(qrs_path, query_store, allow_pickle=True)
#     np.save(ris_level_path, leveled_stores[0], allow_pickle=True)
#     np.save(ris_level_path_enc, leveled_stores[1], allow_pickle=True)
#     np.save(qrs_level_path, leveled_stores[2], allow_pickle=True)
#
# #%%
# (oars50, recalls50, flags, recalls5, recalls10, recalls20, recalls30, recalls40, oars5, oars10, oars20,
#  oars30, oars40, result_sizes, query_sizes) \
#     = (performance_gen
#        (df, query_store, result_store, result_store_enc, isEnc=True, timeCalFlag=True)
#        )
#
# (oars50, recalls50, flags, recalls5, recalls10, recalls20, recalls30, recalls40, oars5, oars10, oars20,
#  oars30, oars40,query_sizes1, query_sizes2, result_sizes1, result_sizes2)  \
#     = (
#     performance_gen_leveled(
#         df, result_store, result_store_enc,
#         query_store_level1=leveled_stores[2],
#         result_store_level1=leveled_stores[0],
#         result_store_level1_enc=leveled_stores[1],
#         isEnc=True, timeCalFlag=True
#     )
# )