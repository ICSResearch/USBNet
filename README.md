## USB-Net-ICS

Put the pth files in the folder "model".

- Test
```
python test.py --model=usbnet --cs_ratio=25
```
The results will be generated in the folder "./results/usbnet/{dataset}/{cs_ratio}/",
where results.csv will save the results in the format "{Image},{PSNR},{SSIM},{Time}".

- Train

1. Multi-GPU
```
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 train.py --model=usbnet --data_path="" --eval_data_path="" --cs_ratio=10 --blr=1e-4 --min_lr=1e-6 --epochs=400 --batch_size=16 --warmup_epochs=10 --input_size=96
```
2. Single GPU
```
python train.py --model=usbnet --data_path="" --eval_data_path="" --cs_ratio=10 --blr=1e-4 --min_lr=1e-6 --epochs=400 --batch_size=16 --warmup_epochs=10 --input_size=96
```

## USB-Net-CSMRI


Put the pth files in the folder "model-Cartesian" or "model-Radial".

- Test
```
python test.py --model=usbnet --cs_ratio=5 --input_size=256 --mask_type=Radial
```
The results will be generated in the folder "./results/usbnet/{dataset}/{cs_ratio}/",
where results.csv will save the results in the format "{Image},{PSNR},{SSIM},{Time}".

- Train

1. Multi-GPU
```
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 train.py --model=usbnet --data_path="" --eval_data_path="" --cs_ratio=5 --blr=5e-5 --min_lr=1e-6 --epochs=100 --batch_size=1 --warmup_epochs=10 --input_size=256 --mask_type=Radial
```
2. Single GPU
```
python train.py --model=usbnet --data_path="" --eval_data_path="" --cs_ratio=5 --blr=5e-5 --min_lr=1e-6 --epochs=100 --batch_size=1 --warmup_epochs=10 --input_size=256 --mask_type=Radial
```

## USB-Net-SCI
### Simulation
Put the pth files in the folder "Sim".
```
python test.py --data_root="path of data"
```
### Real
Put the pth files in the folder "Real".
```
python test_real.py --data_path="path of data" --mask_path="path of mask"
```

## Model
- Model for ICS:
[[BaiduYun](https://pan.baidu.com/s/1_EMg1i-pLq87obCEC3EtdQ?pwd=w6mq)] or 
[[GoogleDrive](https://drive.google.com/drive/folders/1CjDM0wmbFf_TCgosQRVXiMgePAKOoU0F?usp=drive_link)]

- Model for CS-MRI:
[[BaiduYun](https://pan.baidu.com/s/1hjREB8Qh_yWvAJBDLTTVfQ?pwd=w6mq)] or
[[GoogleDrive](https://drive.google.com/drive/folders/1UlWixkmW3YRxObouhy4cQX3UbTjhRnkp?usp=sharing)]/[[GoogleDrive](https://drive.google.com/drive/folders/1FUGob2nl-jstB34NyJa1i7KISBrMUhBC?usp=sharing)]

- Model for SCI:
[[BaiduYun](https://pan.baidu.com/s/1z1d2-GNbxrD0WUFTBwIfsA?pwd=w6mq)]
