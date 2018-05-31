### Four algorithms have been reimplementationed

-  [Deepwalk](https://arxiv.org/pdf/1403.6652.pdf), assess on tencent weibo dataset, AUC: 0.7548
- [LINE](http://www.arxiv.org/pdf/1503.03578.pdf), assess on tencent weibo dataset, AUC: 0.7608
- [Node2Vec](http://www.kdd.org/kdd2016/papers/files/rfp0218-groverA.pdf), assess on tencent weibo dataset, AUC: 0.7553
- [GraRep](https://www.researchgate.net/publication/301417811_GraRep), assess on cora dataset, prediction accuracy: 0.805

### Usage

Go to the ***source*** directory, use following command to run:

``python3 deepwalk_for_tencent.py [or line_for_tencent.py, node2vec_for_tencent.py, grarep_for_cora.py]``

### Requirements

- numpy
- scipy
- networksx
- gensim
- pytorch
- scikit-learn