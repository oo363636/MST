import sysimport multiprocessing as mpimport pandas as pdfrom time import timedef ReadGraph(name):    df = pd.read_csv('sample/' + name + '.csv')    sta, des, weight, E = df.sta.to_list(), df.des.to_list(), df.weight.to_list(), []    for i in range(len(sta) - 1):        E.append((sta[i], des[i], weight[i]))    return sta[-1], Edef Boruvka(N, E, nCPU = None, spark = False):    T, F = set(), Forest(N)    while True:        cheapest_edge = CheapestEdges(E, F)        for r in cheapest_edge:            (u, v, w) = cheapest_edge[r]            if not F.is_on_same_tree(u, v):                T.add((u, v))                F.connect(u, v)            if F.num_trees == 1:                return Tdef Boruvka_Multiprocessing(N, E):    nCPU = mp.cpu_count()    T, F = set(), Forest(N)    pool = mp.Pool(nCPU)    pts = EdgePartition(E, nCPU, F)    while True:        local_cheapest_edges = pool.starmap(CheapestEdges, pts)        cheapest_edges = []        for r in F.roots:            temp = (None, None, sys.maxsize)            for es in local_cheapest_edges:                wt = es[r][2]                if wt < temp[2]:                    temp = es[r]            cheapest_edges.append(temp)        for (u, v, _) in cheapest_edges:            if F.connect(u, v):                T.add((u, v))                if F.num_trees == 1:                    return Tdef Boruvka_Spark(N, E, sc):    EdgeRDD, F, T = sc.parallelize(E), Forest(N), set()    while True:        cheapest_edge = EdgeRDD.filter(lambda e: not F.is_on_same_tree(e[0], e[1])) \                               .flatMap(lambda e: [(F.root(e[0]), e), (F.root(e[1]), e)]) \                               .groupByKey() \                               .map(lambda r: CheapestEdge(r[1])) \                               .collect()        for (u, v, _) in cheapest_edge:            if F.connect(u, v):                T.add((u, v))                if F.num_trees == 1:                    return T    class Forest:    def __init__(self, num_nodes):        self.parent = [[node, 0] for node in range(num_nodes)]        self.num_trees = num_nodes        self.roots = set(range(num_nodes))    def root(self, node):        parent = self.parent[node][0]        if parent == node:            return node        else:            r = self.root(parent)            self.parent[node][0] = r # Path halving.             return r    def is_on_same_tree(self, u, v):        return self.root(u) == self.root(v)    def connect(self, u, v):        ru, rv = self.root(u), self.root(v)        if ru == rv:            return False        self.num_trees -= 1        hu, hv = self.parent[ru][1], self.parent[rv][1]        if hu > hv: # Height Control: Always Connect the Lower tree to the Higher tree.             self.parent[rv][0] = ru            self.roots.remove(rv)        else:            self.parent[ru][0] = rv            self.roots.remove(ru)            if hu == hv:                self.parent[rv][1] += 1        return Truedef CheapestEdge(E):    minw, cheapest_edge = sys.maxsize, None    for edge in E:        w = edge[2]        if w < minw:            minw = w            cheapest_edge = edge    return cheapest_edgedef CheapestEdges(E, F):    cheapest_edges = {r: (None, None, sys.maxsize) for r in F.roots}    for (u, v, w) in E:        if not F.is_on_same_tree(u, v):            ru, rv = F.root(u), F.root(v)            if w < cheapest_edges[ru][2]:                cheapest_edges[ru] = (u, v, w)            if w < cheapest_edges[rv][2]:                cheapest_edges[rv] = (u, v, w)    return cheapest_edgesdef EdgePartition(E, nCPU, F):    num_edges = len(E)    w, r, s, pts = num_edges // nCPU, num_edges % nCPU, 0, []    for i in range(r):        e = s + w + 1        pts.append((E[s:e], F))        s = e    for i in range(nCPU - r):        e = s + w        pts.append((E[s:e], F))        s = e    return pts# Testif __name__ == '__main__':    import findspark    findspark.add_packages("graphframes:graphframes:0.8.0-spark3.0-s_2.12")    findspark.init()    import pyspark    sc = pyspark.SparkContext(appName = "Boruvka")    sc.addPyFile('graphframes-0.8.0-spark3.0-s_2.12.jar')    graph_size, rum_time, run_time_muti, rum_time_spark = [], [], [], []    for size in range(100000, 1000001, 100000):        graph_size.append(size)        print('Graph Size:', size)        N, E = ReadGraph(str(size))        print('Time Cost(s):')        t, T = time(), Boruvka(N, E)        t = time() - t        rum_time.append(t)        print('Single Process:', t)        t, T = time(), Boruvka_Multiprocessing(N, E)        t = time() - t        run_time_muti.append(t)        print('Muti-Processes:', t)        t, T = time(), Boruvka_Spark(N, E, sc)        t = time() - t        rum_time_spark.append(t)        print('Spark:', t, '\n')    sc.stop()    pd.DataFrame({'size': graph_size, 'Python': rum_time, 'Muti': run_time_muti, 'Spark':rum_time_spark}) \      .to_csv('result.csv')