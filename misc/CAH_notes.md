### Notes for things that I (Chris Hamm) have done with the `T-LSTM` neural network started by Baytas *et al.* 2017.


To execute bash script:

1. `source activate py2_ML`
1. `python code/main.py 1 "data/Split0" 1e-3 50 1.0 128 64 'output/final_model'`
1. `python code/main.py 0 "data/Split0" 128 64 'output/final_model'`


**Things done**

2018-09-25:
- Created branch `Hamm_dev`
- Sent code to <kbd>Code/</kbd>
- Sent data to <kbd>Data/</kdb>
- Create a `conda` environment called `py2_ML` and saved to <kbd>misc/</kbd>
- Created a `jupyter notebook` for the `T-LSTM` script using a kernal based on the `py2_ML` environment
- Created branch `dev` from `master`. Merged `Hamm_dev` into `dev` and deleted `Hamm_dev`
- Created branch `T-LSTM-reg` to experiment with changing the softmax layer to a regression. Created `T-LSTM-reg.ipynb`.

2018-09-26:
- Manually manipulating the `tensorflow` code

2018-10-08:
- What does `prev_cell = prev_cell - C_ST + C_ST_dis` represent from their formula?
  - `C_ST` = short term memory
  - `C_ST_dis` = discounted short term memory
- What do `self.Wo` and `self.bo` represent?
