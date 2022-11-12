## 问题描述

在有向图G中，每一边都有一个非负边权。要求图G的从源顶点s到所有其他顶点的最短路径。

## 解题思路
### 分支限界算法

 解单源最短路径问题的优先队列式分支限界法用一极小堆来存储活结点表。其优先级是结点所对应的当前路长。
 
- 从图G的源顶点s和空优先队列开始。
- 结点s成为扩展节点，它的儿子结点被依次插入堆中。
- 从堆中取出具有最小当前路长的结点作为当前扩展结点，并依次检查与当前扩展结点相邻的所有顶点。
- 如从当前扩展结点i到顶点j有边可达，且从源出发，途经顶点i再到顶点j的路径的长度小于当前已得到的s到j的最优路径长度，则将该顶点作为活结点插入到活结点优先队列中。
- 这个结点的扩展过程一直继续到活结点优先队列为空时为止。

### 算法实现

      # 初始化图参数 用字典初始初始化这个图
      G = {1: {2: 4, 3: 2, 4: 5},
           2: {5: 7, 6: 5},
           3: {6: 9},
           4: {5: 2, 7: 7},
           5: {8: 4},
           6: {10: 6},
           7: {9: 3},
           8: {10: 7},
           9: {10: 8},
           10: {}
           }

      inf = 9999
      # 保存源点到各点的距离，为了让顶点和下标一致，前面多了一个inf不用在意。
      length = [inf, 0, inf, inf, inf, inf, inf, inf, inf, inf, inf]
      Q = []


      # FIFO队列实现
      def branch(G, v0):
          Q.append(v0)
          dict = G[1]
          while len(Q) != 0:
              # 队列头元素出队
              head = Q[0]
              # 松弛操作，并且满足条件的后代入队
              for key in dict:
               if length[head] + G[head][key] <= length[key]:
                   length[key] = length[head] + G[head][key]
                   Q.append(key)
           # 松弛完毕，队头出列
           del Q[0]
           if len(Q) != 0:
               dict = G[Q[0]]


     # 优先队列法实现
     def branch(G, v0):
         Q.append(v0)
         while len(Q) != 0:
             min = 99999
             flag = 0
             # 找到队列中距离源点最近的点
             for v in Q:
                 if min > length[v]:
                     min = length[v]
                     flag = v
             head = flag
             dict = G[head]
             # 找到扩散点后进行松弛操作
             for key in dict:
                 if length[head] + G[head][key] <= length[key]:
                     length[key] = length[head] + G[head][key]
                     Q.append(key)
             # 松弛完毕后，该扩散点出队
             Q.remove(head)


     branch(G, 1)
     print(length)

