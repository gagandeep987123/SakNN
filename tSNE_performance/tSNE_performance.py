import numpy as np
import concurrent.futures

from sklearn.utils import axis0_safe_slice


def kNNData(data, point, k):
    distances = np.sqrt(np.sum((point - data) ** 2, axis=1))
    indices = np.argsort(distances)[:k]
    return indices

def OAR(sub_data, org_data, point, k):
    distances_sub = np.sqrt(np.sum((sub_data - point) ** 2, axis=1))
    distances_org = np.sqrt(np.sum((org_data - point) ** 2, axis=1))
    distances_sub.sort()
    distances_org.sort()
    distances_sub = distances_sub[1:k]
    distances_org = distances_org[1:k]
    ratios = distances_sub / distances_org
    return ratios


def task(i, data, embedding_data,
         result_store,
         k=50):
    z1 = kNNData(embedding_data, embedding_data[i], 2000)
    z2 = kNNData(data, data[i], k)


    # top1in_x.append(len(np.intersect1d(z1, z2[:1])))
    for x in [250, 300, 400, 500, 750, 1000, 2000]:
        z3_oar = OAR(data[z1[:x]], data, data[i], k)
        store_key = str(x)
        for n in [2, 3, 4, 5, 10, 20, 30, 40, 50]:
            topn_key = str(n)
            result_store[store_key][topn_key]['ans'].append(len(np.intersect1d(z1[:x], z2[:n])))
            result_store[store_key][topn_key]['oar'].append(np.average(z3_oar[:n-1],axis = 0))


    # if i%100 == 0:
    # print(i, top1in_x[-1],top2in_x[-1],top3in_x[-1],top4in_x[-1],top5in_x[-1],top10in_x[-1],top20in_x[-1],top30in_x[-1],top40in_x[-1],top50in_x[-1])


def performance_test(data, embedding_data, workers=25, query_iterations=1000):
    indices = np.random.permutation(list(range(data.shape[0])))

    result_store = {}
    for x in [250, 300, 400, 500, 750, 1000, 2000]:
        store_key = str(x)
        if store_key not in result_store:
            result_store[store_key] = {}
        for n in [2, 3, 4, 5, 10, 20, 30, 40, 50]:
            topn_key = str(n)
            result_store[store_key][topn_key] = {
                'ans':[],
                'oar':[]
            }

    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        # Submit tasks to the thread pool
        futures = []
        for i in range(query_iterations):
            future = executor.submit(task, indices[i], data, embedding_data,
                                     result_store
                                     )
            futures.append(future)

    print("All tasks completed")

    for store in result_store:
        print("k in higher dimension:", store)
        for n in [2, 3, 4, 5, 10, 20, 30, 40, 50]:
            topn_key = str(n)
            print(np.sum(np.array(result_store[store][topn_key]['ans']) >= n))
        print("\n")

    #recall calculations
    for store in result_store:
        print("k in higher dimension:", store)
        for n in [2, 3, 4, 5, 10, 20, 30, 40, 50]:
            topn_key = str(n)
            arr = np.array(result_store[store][topn_key]['ans'])
            oar = np.average(result_store[store][topn_key]['oar'][:n-1])
            print(n,":",np.sum(arr)/(n*arr.shape[0]), '\toar : ', oar)
        print("\n")


# with open('../tSNE_transform/transformationData/random_2d_scikit.npy', 'rb') as f:
#     embedding = np.load(f)
#     df = np.load(f)
#     performance_test(df, embedding)