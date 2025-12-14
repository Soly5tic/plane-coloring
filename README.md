利用 GNN 搜索平面单位距离图的高色数子图。

- ga.py：一个随机搜索框架，暂时没有接入 GNN。
- udg_builder.py：用于构建平面单位距离图的类，其中对于 Moser Spindle 的实现存在问题。
- graph_coloring.py：用于判断图是否可着色的函数，利用 kissat 求解器。
- gnn_model.py、train_gnn.py：GNN 模型的定义和训练脚本。