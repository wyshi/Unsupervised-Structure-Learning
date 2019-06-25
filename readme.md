## Unsupervised Dialog Structure Learning
Codebase for [Unsupervised Dialog Structure Learning](https://arxiv.org/abs/1904.03736), published as a long paper in NAACL 2019. The codebase is developed based on [NeuralDialog-CVAE](https://github.com/snakeztc/NeuralDialog-CVAE).


If you use the datasets or any source codes included in this repository in your
work, please cite the following paper. The bibtex is listed below:

    @article{shi2019unsupervised,
     title={Unsupervised Dialog Structure Learning},
     author={Shi, Weiyan and Zhao, Tiancheng and Yu, Zhou},
     journal={arXiv preprint arXiv:1904.03736},
     year={2019}
    }
 
    
            
### Requirements
Listed in the requirements.txt. I was using a pretty old tensorflow version when first developing the project. I think most of the major functions are still supported but haven't tested it on the new versions yet.

    python 2
    tensorflow == 1.0.1

            
### Train: 
python main.py --result_path data/results/whatever_name.pkl



### Test: 
python main.py --result_path data/results/whatever_name.pkl --forward_only True --test_path runSomeTimeStamp

After the training, there will be a directory at working/runSomeTimeStamp (e.g. run1532935232), just copy the dir name and pass it to test_path.

#### Interpretation
In interpretation.py.

### Datasets
1. [CamRest676](https://github.com/shawnwun/NNDIAL)
2. [SimDial](https://github.com/snakeztc/SimDial)