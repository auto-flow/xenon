import threading
from collections import defaultdict


class MockEstimator:
    def __init__(self, name):
        self.name = name

    def transform(self, X):
        print(self.name)
        return f"{self.name}({X})"


in_nodes = []
DAG = {
    'scaler': ['num', 'processed'],
    'encoder1': ['cat', 'processed'],
    'encoder2': ['highR_cat', 'processed'],
    '降维': ['processed', 'pre_final'],
    '特征筛选': ['processed', 'pre_final'],
    '拼接': ['pre_final', 'final'],
    '拟合': ['final', 'target'],
}
algo2obj = {k: MockEstimator(k) for k in DAG}

in_order = defaultdict(int)
nodes = set()
for algo_name, (in_node, out_node) in DAG.items():
    in_order[out_node] += 1
    nodes |= {in_node, out_node}

for fg in nodes:
    if in_order[fg] == 0:
        in_nodes.append(fg)

print(in_nodes)
print(in_order)

semaphore = {}

threads = {}
for fg in nodes:
    semaphore[fg] = threading.Semaphore(0)

data_store_ = {k: [k] for k in in_nodes}
data_store = defaultdict(list)
data_store.update(data_store_)


def concat_data(data_list):
    return ",".join(data_list)


class TaskRunner():
    def __init__(self, algo_name, in_node, out_node, data_store, semaphore_map):
        self.out_node, self.in_node, self.semaphore_map = out_node, in_node, semaphore_map
        self.algo_name, self.algo_name = algo_name, data_store

    def __call__(self):
        func = algo2obj[self.algo_name].transform
        for _ in range(in_order[fg]):
            self.semaphore_map[self.in_node].acquire()
        input_data = concat_data(self.data_store[self.in_node])
        output_data = func(input_data)
        self.data_store[self.out_node].append(output_data)
        self.semaphore_map[self.out_node].release()
        return output_data


class Schedule:
    def __init__(self):
        self.threads_map = {}
        self.semaphore_map = {}
        self.data_store = defaultdict(list)
        self.start_nodes = set()
        self.end_node = None

    def register_jobs(self, graph: Graph):
        self.start_nodes = graph.start_nodes
        self.end_node = ...
        for node in graph.nodes:
            self.semaphore_map[node] = threading.Semaphore(0)
        for algo_name, in_node, out_node in graph.edges_list:
            threads[algo_name] = threading.Thread(target=TaskRunner(
                algo_name, in_node, out_node, self.data_store, self.data_store, self.semaphore_map), args=[])

    def submit_jobs(self, data: DataContainer):
        for node in self.start_nodes:
            self.data_store[node].append(data.filter_group(node))
        for thread in threads:
            thread.start()
        for node in self.start_nodes:
            self.semaphore_map[node].release()
        for thread in threads:
            thread.join()
        return self.data_store[self.end_node]


for algo_name, (in_node, out_node) in DAG.items():
    threads[algo_name] = threading.Thread(target=TaskRunner(algo_name, in_node, out_node), args=[])
    threads[algo_name].start()

for in_node in in_nodes:
    semaphore[in_node].release()

for algo_name in DAG:
    threads[algo_name].join()

print(data_store)
