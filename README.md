# SURGE: Sequential Recommendation with Graph Neural Networks

This is our TensorFlow implementation for the paper:

*[Sequential Recommendation with Graph Neural Networks.](https://arxiv.org/pdf/2106.14226.pdf)* SIGIR '21: Proceedings of the 44th International ACM SIGIR Conference on Research and Development in Information Retrieval.

Please cite our paper if you use this repository.
```
@inproceedings{chang2021sequential,
  title={Sequential Recommendation with Graph Neural Networks},
  author={Chang, Jianxin and Gao, Chen and Zheng, Yu and Hui, Yiqun and Niu, Yanan and Song, Yang and Jin, Depeng and Li, Yong},
  booktitle={Proceedings of the 44th International ACM SIGIR Conference on Research and Development in Information Retrieval},
  pages={378--387},
  year={2021}
}
```

The code is tested under a Linux desktop with TensorFlow 1.15.2 and Python 3.6.8.



## Data Pre-processing



The script is `reco_utils/dataset/sequential_reviews.py` which will be automatically excuted when there exists no pre-processed training file.


  

## Model Training

To train our model on `Kuaishou` dataset (with default hyper-parameters): 

```
python examples/00_quick_start/sequential.py --dataset kuaishou
```

or on `Taobao` dataset:

```
python examples/00_quick_start/sequential.py --dataset taobao
``` 

## Misc

The implemention is based on *[Microsoft Recommender](https://github.com/microsoft/recommenders)*.

