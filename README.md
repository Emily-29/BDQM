# BDQM: Blind Dehazed Image Quality Assessment

This is the reference PyTorch implementation for the model using the method described in 
**Blind Dehazed Image Quality Assessment: A Deep CNN-Based Approach**.


The project code is currently being organized and uploaded. Please check back later or follow our updates.




## Installation

Assuming a fresh [Anaconda](https://www.anaconda.com/download/) distribution, you can install the dependencies with:
```shell
conda install pytorch==1.13.0 torchvision==0.14.0 torchaudio==0.13.0 -c pytorch
conda install pytorch-lightning==1.7.7 -c conda-forge
pip install hydra-core
pip install scipy
pip install opencv-contrib-python
```
We ran our experiments with PyTorch 1.13.0, CUDA 11.7, Python 3.7.16.

## Testing

```
python test.py
```

## Paper and License

If you find our work useful in your research please consider citing our paper:

```
@article{10058506,
  author={Lv, Xiao and Xiang, Tao and Yang, Ying and Liu, Hantao},
  title={Blind Dehazed Image Quality Assessment: A Deep CNN-Based Approach}, 
  journal={IEEE Transactions on Multimedia}, 
  year={2023},
  volume={25},
  number={ },
  pages={9410-9424},
  doi={10.1109/TMM.2023.3252267}}
```

The BDQM code is licensed under [MIT License](LICENSE).

