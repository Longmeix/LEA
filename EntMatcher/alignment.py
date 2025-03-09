import time
import gc
import multiprocessing
import numpy as np
from scipy.spatial.distance import cdist


def calculate_rank(idx, sim_mat, top_k, accurate, total_num):
    assert 1 in top_k
    mr = 0
    mrr = 0
    hits = [0] * len(top_k)
    hits1_rest = set()
    for i in range(len(idx)):
        gold = idx[i]
        if accurate:
            rank = (-sim_mat[i, :]).argsort()
        else:
            rank = np.argpartition(-sim_mat[i, :], np.array(top_k) - 1)
        hits1_rest.add((gold, rank[0]))
        assert gold in rank
        rank_index = np.where(rank == gold)[0][0]
        mr += (rank_index + 1)
        mrr += 1 / (rank_index + 1)
        for j in range(len(top_k)):
            if rank_index < top_k[j]:
                hits[j] += 1
    mr /= total_num
    mrr /= total_num
    return mr, mrr, hits, hits1_rest

def task_divide(idx, n):
    total = len(idx)
    if n <= 0 or 0 == total:
        return [idx]
    if n > total:
        return [idx]
    elif n == total:
        return [[i] for i in idx]
    else:
        j = total // n
        tasks = []
        for i in range(0, (n - 1) * j, j):
            tasks.append(idx[i:i + j])
        tasks.append(idx[(n - 1) * j:])
        return tasks

def greedy_alignment(embed1, embed2, top_k, nums_threads, metric, normalize, csls_k, accurate):
    """
        Search alignment with greedy strategy.

        Parameters
        ----------
        embed1 : matrix_like
            An embedding matrix of size n1*d, where n1 is the number of embeddings and d is the dimension.
        embed2 : matrix_like
            An embedding matrix of size n2*d, where n2 is the number of embeddings and d is the dimension.
        top_k : list of integers
            Hits@k metrics for evaluating results.
        nums_threads : int
            The number of threads used to search alignment.
        metric : string
            The distance metric to use. It can be 'cosine', 'euclidean' or 'inner'.
        normalize : bool, true or false.
            Whether to normalize the input embeddings.
        csls_k : int
            K value for csls. If k > 0, enhance the similarity by csls.

        Returns
        -------
        alignment_rest :  list
            pairs of aligned entities
        hits1 : float
            hits@1 values for alignment results
        mr : float
            MR values for alignment results
        mrr : float
            MRR values for alignment results
        """
    t = time.time()
    sim_mat = sim(embed1, embed2, metric=metric, normalize=normalize, csls_k=csls_k)
    num = sim_mat.shape[0]
    if nums_threads > 1:
        hits = [0] * len(top_k)
        mr, mrr = 0, 0
        alignment_rest = set()
        rests = list()
        search_tasks = task_divide(np.array(range(num)), nums_threads)
        pool = multiprocessing.Pool(processes=len(search_tasks))
        for task in search_tasks:
            mat = sim_mat[task, :]
            rests.append(pool.apply_async(calculate_rank, (task, mat, top_k, accurate, num)))
        pool.close()
        pool.join()
        for rest in rests:
            sub_mr, sub_mrr, sub_hits, sub_hits1_rest = rest.get()
            mr += sub_mr
            mrr += sub_mrr
            hits += np.array(sub_hits)
            alignment_rest |= sub_hits1_rest
    else:
        mr, mrr, hits, alignment_rest = calculate_rank(list(range(num)), sim_mat, top_k, accurate, num)
    mrr *= 100
    assert len(alignment_rest) == num
    hits = np.array(hits) / num * 100
    for i in range(len(hits)):
        hits[i] = round(hits[i], 2)
    cost = time.time() - t
    if accurate:
        if csls_k > 0:
            print("accurate results with csls: csls={}, hits@{} = {}%, mr = {:.2f}, mrr = {:.2f}, time = {:.3f} s ".
                  format(csls_k, top_k, hits, mr, mrr, cost))
        else:
            print("accurate results: hits@{} = {}%, mr = {:.2f}, mrr = {:.2f}, time = {:.3f} s ".
                  format(top_k, hits, mr, mrr, cost))
    else:
        if csls_k > 0:
            print("quick results with csls: csls={}, hits@{} = {}%, time = {:.3f} s ".format(csls_k, top_k, hits, cost))
        else:
            print("quick results: hits@{} = {}%, time = {:.3f} s ".format(top_k, hits, cost))
    hits1 = hits[0]
    del sim_mat
    gc.collect()
    return alignment_rest, hits1, mr, mrr

def eval_alignment_by_sim_mat(embeds1, embeds2, top_k=[1,5,10], threads_num=8, mapping=None, metric='inner', normalize=False, csls_k=0, accurate=True):
    if mapping is None:
        alignment_rest_12, hits1_12, mr_12, mrr_12 = greedy_alignment(embeds1, embeds2, top_k, threads_num,
                                                                                metric, normalize, csls_k, accurate)
    else:
        test_embeds1_mapped = np.matmul(embeds1, mapping)
        alignment_rest_12, hits1_12, mr_12, mrr_12 = greedy_alignment(test_embeds1_mapped, embeds2, top_k,
                                                                                threads_num, metric, normalize, csls_k,
                                                                                accurate)
    return alignment_rest_12, hits1_12, mrr_12

def sim(embed1, embed2, metric='inner', normalize=False, csls_k=0):
    """
    Compute pairwise similarity between the two collections of embeddings.

    Parameters
    ----------
    embed1 : numpy
        An embedding matrix of size n1*d, where n1 is the number of embeddings and d is the dimension.
    embed2 : numpy
        An embedding matrix of size n2*d, where n2 is the number of embeddings and d is the dimension.
    metric : str, optional, inner default.
        The distance metric to use. It can be 'cosine', 'euclidean', 'inner'.
    normalize : bool, optional, default false.
        Whether to normalize the input embeddings.
    csls_k : int, optional, 0 by default.
        K value for csls. If k > 0, enhance the similarity by csls.

    Returns
    -------
    sim_mat : numpy
        An similarity matrix of size n1*n2.
    """
    if normalize:
        eval_norm = np.linalg.norm(embed1, axis=1, keepdims=True)
        embed1 = embed1 / eval_norm
        eval_norm = np.linalg.norm(embed2, axis=1, keepdims=True)
        embed2 = embed2 / eval_norm
        
    if metric == 'inner':
        sim_mat = np.matmul(embed1, embed2.T)  # numpy.ndarray, float32
    elif metric == 'cosine' and normalize:
        sim_mat = np.matmul(embed1, embed2.T)  # numpy.ndarray, float32
    elif metric == 'euclidean':
        from sklearn.metrics.pairwise import euclidean_distances
        sim_mat = 1 - euclidean_distances(embed1, embed2)
        print(type(sim_mat), sim_mat.dtype)
        sim_mat = sim_mat.astype(np.float32)
    elif metric == 'cosine':
        sim_mat = 1 - cdist(embed1, embed2, metric='cosine')   # numpy.ndarray, float64
        sim_mat = sim_mat.astype(np.float32)
    elif metric == 'manhattan':
        sim_mat = 1 - cdist(embed1, embed2, metric='cityblock')
        sim_mat = sim_mat.astype(np.float32)
    else:
        sim_mat = 1 - cdist(embed1, embed2, metric=metric)
        sim_mat = sim_mat.astype(np.float32)
    if csls_k > 0:
        sim_mat = csls_sim(sim_mat, csls_k)
    return sim_mat


def csls_sim(sim_mat, k):
    """
    Compute pairwise csls similarity based on the input similarity matrix.

    Parameters
    ----------
    sim_mat : matrix-like
        A pairwise similarity matrix.
    k : int
        The number of nearest neighbors.

    Returns
    -------
    csls_sim_mat : numpy
        A csls similarity matrix of n1*n2.
    """

    nearest_values1 = calculate_nearest_k(sim_mat, k)
    nearest_values2 = calculate_nearest_k(sim_mat.T, k)
    csls_sim_mat = 2 * sim_mat.T - nearest_values1
    csls_sim_mat = csls_sim_mat.T - nearest_values2
    return csls_sim_mat

def calculate_nearest_k(sim_mat, k):
    sorted_mat = -np.partition(-sim_mat, k + 1, axis=1)  # -np.sort(-sim_mat1)
    nearest_k = sorted_mat[:, 0:k]
    return np.mean(nearest_k, axis=1)


# ======================
def cal_csls_sim(sim_mat, k):
    sorted_mat = -np.partition(-sim_mat, k + 1, axis=1)  # -np.sort(-sim_mat1)
    nearest_k = sorted_mat[:, 0:k]
    sim_values = np.mean(nearest_k, axis=1)
    return sim_values


def CSLS_sim(sim_mat1, k, nums_threads):
    tasks = div_list(np.array(range(sim_mat1.shape[0])), nums_threads)
    pool = multiprocessing.Pool(processes=len(tasks))
    reses = list()
    for task in tasks:
        reses.append(pool.apply_async(cal_csls_sim, (sim_mat1[task, :], k)))
    pool.close()
    pool.join()
    sim_values = None
    for res in reses:
        val = res.get()
        if sim_values is None:
            sim_values = val
        else:
            sim_values = np.append(sim_values, val)
    assert sim_values.shape[0] == sim_mat1.shape[0]
    return sim_values

def sim_handler(embed1, embed2, k, nums_threads):
    sim_mat = np.matmul(embed1, embed2.T)
    if k <= 0:
        print("k = 0")
        return sim_mat
    csls1 = CSLS_sim(sim_mat, k, nums_threads)
    csls2 = CSLS_sim(sim_mat.T, k, nums_threads)
    csls_sim_mat = 2 * sim_mat.T - csls1
    csls_sim_mat = csls_sim_mat.T - csls2
    del sim_mat
    gc.collect()
    return csls_sim_mat

def generate_candidates_by_sim_mat(embed1, embed2, dev_alignments, cand_num=20, csls=10, num_thread=16):
    '''
    Output: {ent_id_1:{'ground_rank': rank, 'candidates': candidates}}
        - ground_rank : the rank of entity2 in entity_list2 which matches ent_id_1, according to reference pairs
        - candidates  : top K entities which matches ent_id_1, according to sim_mat = dot(embed1, embed2)
    '''
    ent1, ent2 = [], []
    for anc1, anc2 in dev_alignments:
        ent1.append(anc1)
        ent2.append(anc2)
    ent_idx_1 = {e:i for i, e in enumerate(ent1)}
    ent_frags = task_divide(np.array(ent1), num_thread)
    sim_mat = sim_handler(embed1, embed2, csls, num_thread)
    ref_num = sim_mat.shape[0]
    tasks = task_divide(np.array(range(ref_num)), num_thread)

    pool = multiprocessing.Pool(processes=len(tasks))
    results = []
    for i, task in enumerate(tasks):
        results.append(pool.apply_async(find_candidates_by_sim_mat, args=(ent_frags[i], ent_idx_1, sim_mat[task, :], np.array(ent2), cand_num)))
    pool.close()
    pool.join()

    dic = {}
    for res in results:
        dic = {**dic, **res.get()}
    del results
    gc.collect()
    return dic

def find_candidates_by_sim_mat(frags, ent_idx, sim, entity_list2, k):
    dic = {}
    for i in range(sim.shape[0]):
        ref = ent_idx[frags[i]]
        rank = (-sim[i, :]).argsort()
        rank_index = np.where(rank == ref)[0][0]
        cand_index = np.argpartition(-sim[i, :], k)[:k]
        candidates = entity_list2[cand_index].tolist()
        cand_sims = [float(s) for s in sim[i, cand_index]]
        # min_s, max_s = min(cand_sims), max(cand_sims)
        # cand_sims = [(s - min_s) / (max_s - min_s) for s in cand_sims]
        
        sorted_cand = sorted([(candidates[j], cand_sims[j]) for j in range(k)], key=lambda x: x[1], reverse=True)
        candidates = [c_idx for c_idx, _ in sorted_cand]
        cand_sims  = [c_sim for _, c_sim in sorted_cand]

        dic[int(frags[i])] = {'ref': int(entity_list2[ref]), 'ground_rank':int(rank_index), 
                              'candidates':candidates, 'cand_sims':cand_sims}
    return dic