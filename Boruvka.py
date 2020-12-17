import sys
from psutil import cpu_count
import pandas as pd
from time import time

def ReadGraph(name):
    df = pd.read_csv('sample/' + name + '.csv')
    sta, des, weight, E = df.sta.to_list(), df.des.to_list(), df.weight.to_list(), []
    for i in range(len(sta) - 1):
        E.append((sta[i], des[i], weight[i]))
    return sta[-1], E

def Boruvka(N, E):
    T, parent, height, roots, nt = [], list(range(N)), [0] * N, set(range(N)), N
    while True:
        cheapest_edge = CheapestEdges(E, parent, roots)
        for r in cheapest_edge:
            (u, v, _) = cheapest_edge[r]
            if connect(parent, height, roots, u, v):
                T.append((u, v))
                nt -= 1
                if nt == 1:
                    return T

def BoruvkaSpark(N, E, sc, np):
    EdgeRDD, T, parent, height, roots, nt = sc.parallelize(E, np), [], list(range(N)), [0] * N, set(range(N)), N
    while True:
        cheapest_edge = EdgeRDD.mapPartitions(CheapestEdgesIter(parent, roots)) \
                               .groupByKey() \
                               .mapPartitions(CheapestEdge) \
                               .collect()
                                       
        for (u, v, _) in cheapest_edge:
            if connect(parent, height, roots, u, v):
                T.append((u, v))
                nt -= 1
                if nt == 1:
                    return T

def root(parent, node):
    father = parent[node]
    if father == node:
        return node
    else:
        r = root(parent, father)
        parent[node] = r
        return r

def connect(parent, height, roots, u, v):
    ru, rv = root(parent, u), root(parent, v)
    if ru == rv:
        return False
    hu, hv = height[ru], height[rv]
    if hu > hv: 
        parent[rv] = ru
        roots.remove(rv)
    else:
        parent[ru] = rv
        roots.remove(ru)
        if hu == hv:
            height[rv] += 1
    return True

def CheapestEdge(rs):
    for (r, es) in rs:
        minw = sys.maxsize
        for edge in es:
            w = edge[2]
            if w < minw:
                minw = w
                cheapest_edge = edge
        yield cheapest_edge

def CheapestEdges(E, parent, roots):
    cheapest_edges = {r: (None, None, sys.maxsize) for r in roots}
    for (u, v, w) in E:
        ru, rv = root(parent, u), root(parent, v)
        if ru != rv:
            if w < cheapest_edges[ru][2]:
                cheapest_edges[ru] = (u, v, w)
            if w < cheapest_edges[rv][2]:
                cheapest_edges[rv] = (u, v, w)
    return cheapest_edges

def CheapestEdgesIter(parent, roots):
    def cheapestEdges(E):
        cheapest_edges = CheapestEdges(E, parent, roots)
        for r in cheapest_edges:
            yield (r, cheapest_edges[r])
    return cheapestEdges

# Test
import findspark
findspark.init()
import pyspark
num_CPU = cpu_count(False)
graph_size, py, spark = [], [], [[] for i in range(num_CPU)]
sc = pyspark.SparkContext(appName = "Boruvka")
for size in range(100000, 1000001, 100000):
    graph_size.append(size)
    N, E = ReadGraph(str(size))
    t, T = time(), Boruvka(N, E)
    py.append(time() - t)
    for num_partiton in range(1, 1 + num_CPU):
        t, T = time(), BoruvkaSpark(N, E, sc, np = num_partiton)
        spark[num_partiton - 1].append(time() - t)
sc.stop()
result = {'size': graph_size, 'Python': py}
for i in range(1, 1 + num_CPU):
    result[i] = spark[i - 1]
pd.DataFrame(result).to_csv('result.csv')
