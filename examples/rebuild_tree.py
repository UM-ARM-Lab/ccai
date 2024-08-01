paths = [
    ['pregrasp', 'index', 'turn', 'turn', 'turn', 'turn', 'turn'],
    ['pregrasp', 'turn', 'index', 'turn', 'turn', 'index', 'thumb_middle'],
    ['pregrasp', 'turn', 'thumb_middle', 'index', 'turn', 'turn', 'index'],
    ['pregrasp', 'thumb_middle', 'thumb_middle', 'turn', 'turn', 'turn', 'turn'],
    ['pregrasp', 'thumb_middle', 'index', 'turn', 'turn', 'turn', 'turn'],
    ['pregrasp', 'index', 'thumb_middle', 'thumb_middle', 'turn', 'turn', 'thumb_middle'],
    ['pregrasp', 'thumb_middle', 'turn', 'turn', 'index', 'turn'],
    ['pregrasp', 'thumb_middle', 'turn', 'turn', 'index', 'index'],
    ['pregrasp', 'thumb_middle', 'turn', 'turn', 'turn', 'turn', 'turn'],
    ['pregrasp', 'thumb_middle', 'turn', 'thumb_middle', 'turn', 'thumb_middle', 'turn']
]

new_paths = []
for path in paths:
    for i in range(1, len(path)):
        new_paths.append([path[0]] + path[i:])

new_paths_filtered = list(set([tuple(path) for path in new_paths]))
print(len(new_paths), len(new_paths_filtered))
successors = {}

for path in new_paths_filtered:
    for i in range(1, len(path)-1):
        if path[:i] not in successors:
            successors[path[:i]] = set()
        successors[path[:i]].add(path[:i+1])

import networkx as nx

G = nx.DiGraph()
for key, value in successors.items():
    for v in value:
        G.add_edge(key, v)

# Check if tree
print(nx.is_tree(G))