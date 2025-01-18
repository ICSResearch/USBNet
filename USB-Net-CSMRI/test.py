#%%
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import torchvision
import numpy as np
import glob
from time import time
import cv2
import torch
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
import argparse
from timm.models import create_model
import warnings
from scipy import io as scio
from USBNet import load_sampling_matrix
from USBNet import *
warnings.filterwarnings("ignore")


def main(test_name, args):
    args = parser.parse_known_args()[0]

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = create_model(args.model, ratio=args.cs_ratio, pretrained=True, mask_type=args.mask_type)

    model = torch.nn.DataParallel(model)
    model = model.to(device)
    mask = load_sampling_matrix(args.cs_ratio, args.input_size, args.mask_type).to(device)

    ext = {'/*'}
    filepaths = []
    test_dir = f"../data/dataset/CSMRI/{test_name}/"

    for img_type in ext:
        filepaths = filepaths + glob.glob(test_dir + img_type)

    result_dir = os.path.join(args.result_dir, args.model, test_name, str(args.cs_ratio))
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    ImgNum = len(filepaths)
    PSNR_All, SSIM_All, Time_All = [], [], []

    with torch.no_grad():
        print("\nCS Reconstruction Start")
        for img_no in range(ImgNum):
            imgName = filepaths[img_no]
            # Img = np.abs(scio.loadmat(imgName)['data'], dtype=np.float64)
            Img = cv2.imread(imgName, 0) / 255.

            Img = torch.from_numpy(Img).float()
            Img = torch.unsqueeze(Img, dim=0)
            augs = torchvision.transforms.Compose([
                torchvision.transforms.CenterCrop(args.input_size),
            ])
            Img = augs(Img)
            Img = torch.squeeze(Img, dim=0)
            Iorg = Img.numpy()

            batch_x = torch.Tensor(Img).to(device)
            batch_x = batch_x.unsqueeze(0).unsqueeze(0)
            start = time()
            x_output = model(batch_x, mask)
            end = time()

            x_output = x_output.squeeze(0).squeeze(0)
            Prediction_value = x_output.cpu().data.numpy()
            X_rec = np.clip(Prediction_value, 0, 1)

            rec_PSNR = psnr(Iorg, X_rec, data_range=1.)
            rec_SSIM = ssim(X_rec, Iorg, data_range=1.)

            test_name_split = os.path.split(imgName)
            print("[%02d/%02d] Run time for %s is %.4f, PSNR is %.2f, SSIM is %.4f" % (
                img_no, ImgNum, test_name_split[1], (end - start), rec_PSNR, rec_SSIM))

            im_rec_rgb = np.array(X_rec * 255)
            im_rec_rgb = np.clip(im_rec_rgb, 0, 255).astype(np.uint8)
            resultName = "./%s/%s" % (result_dir, test_name_split[1])
            with open(os.path.join(result_dir, 'results.csv'), 'a+') as f:
                store_info = f"{resultName},{rec_PSNR},{rec_SSIM},{end - start}\n"
                f.write(store_info)
            cv2.imwrite("%s_ratio_%.2f_PSNR_%.2f_SSIM_%.4f.png" % (resultName, args.cs_ratio, rec_PSNR, rec_SSIM), im_rec_rgb)
            del x_output

            PSNR_All.append(rec_PSNR)
            SSIM_All.append(rec_SSIM)
            Time_All.append(end - start)

    print('\n')
    output_data = "CS ratio is %.2f, Avg PSNR/SSIM/Time for %s is %.2f/%.4f/%.4f"% (
        args.cs_ratio, test_name, np.mean(PSNR_All), np.mean(SSIM_All), np.mean(Time_All))
    print(output_data)
    with open(os.path.join(result_dir, 'results.csv'), 'a+') as f:
        store_info = f"avg, {np.mean(PSNR_All)}, {np.mean(SSIM_All)}, {np.mean(Time_All)}\n"
        f.write(store_info)
    print("CS Reconstruction End")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cs_ratio', type=int, default=5, help='set sensing rate')
    parser.add_argument('--input_size', type=int, default=256, help='input size (default: 256)')

    parser.add_argument('--model', type=str, default='usbnet', help='model name')
    parser.add_argument('--block_size', type=int, default=32, help='block size (default: 32)')
    parser.add_argument('--result_dir', type=str, default='results', help='result directory')
    
    parser.add_argument('--mask_type', type=str, default='Radial', help='mask type (default: Radial)')
    for test_name in ["brain_test"]:
        main(test_name, parser)


