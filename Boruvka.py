import sysimport pandas as pdfrom time import timedef ReadGraph(name):    df = pd.read_csv('sample/' + name + '.csv')    sta, des, weight, E = df.sta.to_list(), df.des.to_list(), df.weight.to_list(), []    for i in range(len(sta) - 1):        E.append((sta[i], des[i], weight[i]))    return sta[-1], Edef Boruvka(N, E):    T, parent, height, roots, nt = [], list(range(N)), [0] * N, set(range(N)), N    while True:        cheapest_edge = CheapestEdges(E, parent, roots)        for r in cheapest_edge:            (u, v, _) = cheapest_edge[r]            if connect(parent, height, roots, u, v):                T.append((u, v))                nt -= 1                if nt == 1:                    return Tdef Boruvka_Spark(N, E, sc):    EdgeRDD, T, parent, height, roots, nt = sc.parallelize(E), [], list(range(N)), [0] * N, set(range(N)), N    while True:        cheapest_edge = EdgeRDD.mapPartitions(CheapestEdgesIter(parent, roots)) \                               .groupByKey() \                               .map(lambda x: CheapestEdge(x[1])) \                               .collect()        for (u, v, _) in cheapest_edge:            if connect(parent, height, roots, u, v):                T.append((u, v))                nt -= 1                if nt == 1:                    return Tdef root(parent, node):    father = parent[node]    if father == node:        return node    else:        r = root(parent, father)        parent[node] = r        return rdef connect(parent, height, roots, u, v):    ru, rv = root(parent, u), root(parent, v)    if ru == rv:        return False    hu, hv = height[ru], height[rv]    if hu > hv:         parent[rv] = ru        roots.remove(rv)    else:        parent[ru] = rv        roots.remove(ru)        if hu == hv:            height[rv] += 1    return Truedef CheapestEdge(E):    minw = sys.maxsize    for edge in E:        w = edge[2]        if w < minw:            minw = w            cheapest_edge = edge    return cheapest_edgedef CheapestEdges(E, parent, roots):    cheapest_edges = {r: (None, None, sys.maxsize) for r in roots}    for (u, v, w) in E:        ru, rv = root(parent, u), root(parent, v)        if ru != rv:            if w < cheapest_edges[ru][2]:                cheapest_edges[ru] = (u, v, w)            if w < cheapest_edges[rv][2]:                cheapest_edges[rv] = (u, v, w)    return cheapest_edgesdef CheapestEdgesIter(parent, roots):    def cheapestEdges(E):        cheapest_edges = CheapestEdges(E, parent, roots)        for r in cheapest_edges:            yield (r, cheapest_edges[r])    return cheapestEdges# Testif __name__ == '__main__':    import findspark    findspark.add_packages("graphframes:graphframes:0.8.0-spark3.0-s_2.12")    findspark.init()    import pyspark    sc = pyspark.SparkContext(appName = "Boruvka")    sc.addPyFile('graphframes-0.8.0-spark3.0-s_2.12.jar')    graph_size, py, spark = [], [], []    for size in range(100000, 1000001, 100000):        graph_size.append(size)        print('Graph Size:', size)        N, E = ReadGraph(str(size))        print('Time Cost(s):')        t, T = time(), Boruvka(N, E)        t = time() - t        py.append(t)        print('Single Process:', t)        t, T = time(), Boruvka_Spark(N, E, sc)        t = time() - t        spark.append(t)        print('Spark:', t, '\n')    sc.stop()    pd.DataFrame({'size': graph_size, 'Python': py, 'Spark':spark}).to_csv('result.csv')