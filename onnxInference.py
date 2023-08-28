import torch.onnx
import torch
from PIL import Image
from torchvision import transforms
import torch.nn.functional as F
import numpy as np
from matplotlib import pyplot as plt
import torch.onnx
import onnxruntime
import os
import glob

class Args:
    def __init__(self):
        self.img_dir = None
        self.img_folder_dir = 'onnx/3'
args = Args()

def inference_onnx_model(img):
    ort_session = onnxruntime.InferenceSession('onnx/Lapdepth_0821.MobileNetV2.onnx')
    # numpy iput
    ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(img)}
    ort_outs = ort_session.run(None, ort_inputs)
    ort_out = ort_outs[0]
    out = torch.from_numpy(ort_out).float()
    return out

def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

def show_depth_image(out, out_flip):
    out_flip = torch.flip(out_flip, [3])
    out = 0.5 * (out + out_flip)
    out = out[0, 0]
    out = out * 1000.0
    out = out.cpu().detach().numpy().astype(np.uint16)
    out = (out / out.max()) * 255.0
    return out

## Inference
if args.img_dir is not None:
    if args.img_dir[-1] == '/':
        args.img_dir = args.img_dir[:-1]
    img_list = [args.img_dir]
    result_filelist = ['./out_' + args.img_dir.split('/')[-1]]
elif args.img_folder_dir is not None:
    if args.img_folder_dir[-1] == '/':
        args.img_folder_dir = args.img_folder_dir[:-1]
    png_img_list = glob.glob(args.img_folder_dir + '/*.png')
    jpg_img_list = glob.glob(args.img_folder_dir + '/*.jpg')
    img_list = png_img_list + jpg_img_list
    img_list = sorted(img_list)
    result_folder = './onnx/3/MobileNetV2'
    if not os.path.exists(result_folder):
        os.makedirs(result_folder)
    result_filelist = []
    for file in img_list:
        result_filename = result_folder + '/MobileNetV2_' + file.split('/')[-1]
        result_filelist.append(result_filename)

print("=> process..")
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

for i, img_file in enumerate(img_list):
    image = Image.open(img_file)
    image = np.asarray(image, dtype=np.float32) / 255.0
    if image.ndim == 2:
        image = np.expand_dims(image, 2)
        image = np.repeat(image, 3, 2)
    image = image.transpose((2, 0, 1))
    # numpy转换成tensor
    image = torch.from_numpy(image).float()
    image = normalize(image)
    image = image.cuda()
    _, org_h, org_w = image.shape
    image = image.unsqueeze(0)
    new_h = 480
    new_w = 640
    new_w = int((new_w // 16) * 16)
    image = F.interpolate(image, (new_h, new_w), mode='bilinear')

    img_flip = torch.flip(image,[3])
    with torch.no_grad():
        out = inference_onnx_model(image)
        out_flip = inference_onnx_model(img_flip)
    out = show_depth_image(out, out_flip)
    result_filename = result_filelist[i]
    plt.imsave(result_filename, np.log10(out), cmap='plasma_r')
    if (i + 1) % 5 == 0:
        print("=>", i + 1, "th image is processed..")
print("=> Done.")
