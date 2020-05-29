This repository tests Lookahead optimisers at the Penn Treebank dataset using the code from 'Regularizing and optimising LSTM language models'.

Penn Treebank dataset:
@article{marcus1993building,
  title={Building a large annotated corpus of English: The Penn Treebank},
  author={Marcus, Mitchell and Santorini, Beatrice and Marcinkiewicz, Mary Ann},
  year={1993}
}
------------------------------------------------------------------------------

Regularizing and optimising LSTM language models (original code):
@article{merity2017regularizing,
  title={Regularizing and optimizing LSTM language models},
  author={Merity, Stephen and Keskar, Nitish Shirish and Socher, Richard},
  journal={arXiv preprint arXiv:1708.02182},
  year={2017}
}

We used the original code from: https://github.com/salesforce/awd-lstm-lm
------------------------------------------------------------------------------

Lookahead Optimizer: k steps forward, 1 step back:
@article{zhang2019lookahead,
  title={Lookahead Optimizer: k steps forward, 1 step back},
  author={Zhang, Michael R and Lucas, James and Hinton, Geoffrey and Ba, Jimmy},
  journal={arXiv preprint arXiv:1907.08610},
  year={2019}
}

We added the optimizer code code from: https://github.com/michaelrzhang/lookahead
--------------------------------------------------------------------------------

- A few changes were done to the original code to be compatible with newer versions of pytorch(1.4.0)
- Added on main.py the functionality of using the Lookahead optimizers (la_adam, la_sgd) the baseline ones (adam, sgd) and kept the original (nt_asgd) optimizer
- Added the functionality of decreasing the learning rate by half if validation perplexity does not improve for 15 epochs as descibed in 'Lookahead Optimizer: k steps forward, 1 step back'
- Calculating the running validation loss as weighted over the size of batches (those values are reported with the accompaniedpaper), in contrast to the original code
