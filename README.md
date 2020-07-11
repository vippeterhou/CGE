# CGE
This repo is related to the paper:

[**Customized Graph Embedding: Tailoring Embedding Vectors to different Applications**](https://arxiv.org/pdf/1911.09454v3.pdf)

Welcome to check it out and cite it.

## Dependencies
Here is the version of some majoy dependencies:
```bash
python=2.7
theano=1.0.3
lasagne=0.2
numpy=1.15.4
```

## Dataset
We include CITESEER, CORA, PUBMED in the `data` folder.

## Quick Start
You can start with default settings.

For transductive setting:
```bash
sh run_trans_main.sh
```
For inductive setting:
```bash
sh run_ind_main.sh
```

You may modify the `trans_main.py` and `ind_main.py` for different settings.

## Citation
Please cite it if it helps your research:

    @misc{hou2019customized,
        title={Customized Graph Embedding: Tailoring Embedding Vectors to different Applications},
        author={Bitan Hou and Yujing Wang and Ming Zeng and Shan Jiang and Ole J. Mengshoel and Yunhai Tong and Jing Bai},
        year={2019},
        eprint={1911.09454},
        archivePrefix={arXiv},
        primaryClass={cs.LG}
    }
