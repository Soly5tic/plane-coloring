利用 GNN 搜索平面单位距离图的高色数子图。

- ga.py：一个随机搜索框架，暂时没有接入 GNN。
  - 现在 update_fitness 函数中注释掉的部分是使用 GNN 的启发式函数。
  - 在点数超过上限且无法进行 k-core 剪枝时，进行了删除距离原点最远的点的操作。实际上这个思路很不好，需要寻求改进。
- udg_builder.py：用于构建平面单位距离图的类，其中对于 Moser Spindle 的实现存在问题。
- graph_coloring.py：用于判断图是否可着色的函数，利用 kissat 求解器。
- gnn_model.py、train_gnn.py：GNN 模型的定义和训练脚本。