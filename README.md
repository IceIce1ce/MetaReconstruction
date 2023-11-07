This is the official repository of 

**MetaReconstruction: A Unified Framework for Reconstruction-based Video Anomaly Detection.**

## Setup
```bash
conda create -n meta_reconstruction python=3.10
conda activate meta_reconstruction
conda install pytorch==1.11.0 torchvision==0.12.0 torchaudio==0.11.0 cudatoolkit=11.3 -c pytorch
pip install -r requirements.txt
```

## Dataset Preparation
Download the [CUHK Avenue](https://drive.google.com/file/d/1q3NBWICMfBPHWQexceKfNZBgUoKzHL-i/view), [UCSD Ped2](https://drive.google.com/file/d/1w1yNBVonKDAp8uxw3idQkUr-a9Gj8yu1/view) and [ShanghaiTech](https://drive.google.com/file/d/1rE1AM11GARgGKf4tXb2fSqhn_sX46WKn/view) datasets and structure the data as follows:
```
dataset/
  avenue
    training
      frames
        01
          .jpg
        02
          .jpg
        ...
    testing
      frames
        01
          .jpg
        02
          .jpg
        ...
    avenue.mat
  ped2
    training
      frames
        01
          .jpg
        02
          .jpg
        ...
    testing
      frames
        01
          .jpg
        02
          .jpg
        ...
    avenue.mat
  shanghaitech
    training
      videos
        .avi
    testing
      frames
        01_0014
          .jpg
        01_0015
          .jpg
        ...
      test_frame_mask
        .npy
      test_pixel_mask
        .npy
```

## Usage
To use our model, follow the code snippet below:
```bash
cd Reconstructed_based

# Train, Test and Demo 3D AutoEncoder
bash scripts/train_3dae.sh
bash scripts/eval_3dae.sh
bash demo.sh

# Train, Test and Demo STEAL
bash scripts/train_steal.sh
bash scripts/eval_steal.sh
bash demo.sh

# Train, Test and Demo MemAE
bash scripts/train_memae3d.sh
bash scripts/eval_memae3d.sh
bash demo.sh

# Train, Test and Demo Reconstruction MNAD
bash scripts/train_rmnad.sh
bash scripts/eval_rmnad.sh
bash scripts/demo.sh

# Train, Test and Demo Future Frame Prediction MNAD
bash scripts/train_pmnad.sh
bash scripts/eval_pmnad.sh
bash scripts/demo.sh
```

## MetaReconstruction Model Zoo
TBA.

## Citation
If you find our work useful, please cite the following:
```
@misc{Chi2023,
  author       = {Chi Tran},
  title        = {MetaReconstruction: A Unified Framework for Reconstruction-based Video Anomaly Detection},
  publisher    = {GitHub},
  booktitle    = {GitHub repository},
  howpublished = {https://github.com/IceIce1ce/MetaReconstruction},
  year         = {2023}
}
```

## Contact
If you have any questions, feel free to contact `Chi Tran` 
([ctran743@gmail.com](ctran743@gmail.com)).

##  Acknowledgement
Our framework is built using multiple open source, thanks for their great contributions.
<!--ts-->
* [aseuteurideu/STEAL](https://github.com/aseuteurideu/STEAL)
* [cvlab-yonsei/MNAD](https://github.com/cvlab-yonsei/MNAD)
* [donggong1/memae-anomaly-detection](https://github.com/donggong1/memae-anomaly-detection)
<!--te-->
