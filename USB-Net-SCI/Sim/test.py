# %%
from model import *
from utils.sci_utils import *
import scipy.io as scio
import torch
import os
import numpy as np
from timm.models import create_model
import argparse

parser = argparse.ArgumentParser(
    description="HyperSpectral Image Reconstruction Toolbox"
)
# Hardware specifications
parser.add_argument("--gpu_id", type=str, default="0")

# Data specifications
parser.add_argument(
    "--data_root", type=str, help="dataset directory"
)

# Saving specifications
parser.add_argument("--outf", type=str, default="./result/usbnet/", help="saving_path")

# Model specifications
parser.add_argument("--model", type=str, default="usbnet", help="method name")
parser.add_argument(
    "--input_setting",
    type=str,
    default="Y",
    help="the input measurement of the network: H, HM or Y",
)
parser.add_argument(
    "--input_mask",
    type=str,
    default="Phi_PhiPhiT",
    help="the input mask of the network: Phi, Phi_PhiPhiT or None",
)

opt = parser.parse_known_args()[0]

opt.mask_path = f"{opt.data_root}/TSA_simu_data/"
opt.test_path = f"{opt.data_root}/TSA_simu_data/Truth/"

for arg in vars(opt):
    if vars(opt)[arg] == "True":
        vars(opt)[arg] = True
    elif vars(opt)[arg] == "False":
        vars(opt)[arg] = False

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu_id
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
if not torch.cuda.is_available():
    raise Exception("NO GPU!")


# Intialize mask
mask3d_batch, input_mask = init_mask(opt.mask_path, opt.input_mask, 10)

if not os.path.exists(opt.outf):
    os.makedirs(opt.outf)


def test(model):
    psnr_list, ssim_list = [], []
    test_data = LoadTest(opt.test_path)
    test_gt = test_data.cuda().float()
    input_meas = init_meas(test_gt, mask3d_batch, opt.input_setting)
    model.eval()
    with torch.no_grad():
        model_out = model(input_meas, input_mask)

    for k in range(test_gt.shape[0]):
        psnr_val = torch_psnr(model_out[k, :, :, :], test_gt[k, :, :, :])
        ssim_val = torch_ssim(model_out[k, :, :, :], test_gt[k, :, :, :])
        psnr_list.append(psnr_val.detach().cpu().numpy())
        ssim_list.append(ssim_val.detach().cpu().numpy())
        print(f"{k+1},{psnr_val:.2f},{ssim_val:.4f}")

    pred = np.transpose(model_out.detach().cpu().numpy(), (0, 2, 3, 1)).astype(
        np.float32
    )
    truth = np.transpose(test_gt.cpu().numpy(), (0, 2, 3, 1)).astype(np.float32)
    mea = np.transpose(input_meas.cpu().numpy(), (1, 2, 0)).astype(np.float32)
    psnr_mean = np.mean(np.asarray(psnr_list))
    ssim_mean = np.mean(np.asarray(ssim_list))
    print(f"Avg.,{psnr_mean:.2f},{ssim_mean:.4f}")
    return pred, truth, mea, psnr_list, ssim_list


def results(path):
    mat = scio.loadmat(path)
    method = path.split("/")[-1].split("\\")[-1].split(".")[0]
    with open("results.csv", "a+") as f:
        f.write(f"{method},,\n")
    truth, pred = mat["truth"], mat["pred"]
    truth = np.transpose(truth, (0, 3, 1, 2)).astype(np.float32)
    pred = np.transpose(pred, (0, 3, 1, 2)).astype(np.float32)
    print(truth.shape)

    psnrs, ssims = [], []
    for k in range(pred.shape[0]):
        psnr_val = torch_psnr(
            torch.from_numpy(truth[k, :, :, :]), torch.from_numpy(pred[k, :, :, :])
        )
        ssim_val = torch_ssim(
            torch.from_numpy(truth[k, :, :, :]), torch.from_numpy(pred[k, :, :, :])
        )
        psnrs.append(psnr_val.detach().cpu().numpy())
        ssims.append(ssim_val.detach().cpu().numpy())
        with open("results.csv", "a+") as f:
            f.write(f"{k+1},{psnr_val:.2f},{ssim_val:.4f}\n")
        print(f"{k+1},{psnr_val:.2f},{ssim_val:.4f}")
    with open("results.csv", "a+") as f:
        f.write(f"Avg.,{np.average(psnrs):.2f},{np.average(ssims):.4f}\n")
    print(f"Avg.,{np.average(psnrs):.2f},{np.average(ssims):.4f}")


def main():
    # model
    device = torch.device("cuda")
    model = create_model("usbnet", pretrained=True).to(device)
    pred, truth, mea, psnr_list, ssim_list = test(model)
    name = opt.outf + "Test_result.mat"
    print(f"Save reconstructed HSIs as {name}.")
    scio.savemat(
        name,
        {
            "truth": truth,
            "pred": pred,
            "mea": mea,
            "psnr_list": psnr_list,
            "ssim_list": ssim_list,
        },
    )


if __name__ == "__main__":
    main()
    # path = "./result/usbnet/Test_result.mat"
    # results(path)
