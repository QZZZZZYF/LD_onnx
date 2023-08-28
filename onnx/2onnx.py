from model import LDRN
import torch.onnx
import onnx
import torch
from PIL import Image
from torchvision import transforms
import torch.nn.functional as F
import numpy as np
from matplotlib import pyplot as plt
import torch.onnx
import onnxruntime
import time

class Args:
    def __init__(self):
        self.model_dir = 'pretrained/NYU_LDRN_DenseNet161_epoch35/epoch_30_loss_0.4639_1.pkl'
        self.img_dir = 'example/nyu_demo.jpg'
        self.img_folder_dir = None
        self.seed = 0
        self.encoder = "DenseNet161"
        self.pretrained = "NYU"
        self.norm = "BN"
        self.n_Group = 32
        self.reduction = 16
        self.act = "ReLU"
        self.max_depth = 10
        self.lv6 = False
        self.cuda = True
        self.gpu_num = "0, 1, 2, 3"
        self.rank = 0
        self.input_height = 480
        self.input_width = 640
args = Args()

## TO ONNX
print('=> loading model..')
Model = LDRN(args)
Model = Model.cuda()
Model = torch.nn.DataParallel(Model)
assert (args.model_dir != ''), "Expected pretrained model directory"
Model.load_state_dict(torch.load(args.model_dir), False)
Model.eval()

# 模型输入的维度
image = torch.randn(1, 3, 480, 640).cuda()
onnx_model_path = 'onnx/Lapdepth_0821.DenseNet161.onnx'

torch.onnx.export(
    Model.module,  # 转换的模型
    image,  # 输入的维度
    onnx_model_path,  # 导出的 ONNX 文件名
    export_params=True,  # store the trained parameter weights inside the model file
    # verbose=True,
    opset_version=11,  # ONNX 算子集的版本
    input_names=["image"],  # 输入的 tensor名称，可变
    output_names=["depth"]  # 输出的 tensor名称，可变
    )
print("sucessful")

## check onnx
onnx_model = onnx.load("onnx/Lapdepth_0821.DenseNet161.onnx")
onnx.checker.check_model(onnx_model)
print(onnx.helper.printable_graph(onnx_model.graph))

######################## Inference_img_dir
def preprocess_image(image_path):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    image = Image.open(image_path)
    image = np.asarray(image, dtype=np.float32) / 255.0
    if image.ndim == 2:
        image = np.expand_dims(image,2)
        image = np.repeat(image,3,2)
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
    return image

def inference_onnx_model(img):
    ort_session = onnxruntime.InferenceSession('onnx/Lapdepth_0821.DenseNet161.onnx')
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

start_time = time.time() #增加计时器

image_path = 'example/nyu_demo.jpg'
img = preprocess_image(image_path)
img_flip = torch.flip(img,[3])
out = inference_onnx_model(img)
out_flip = inference_onnx_model(img_flip)
out = show_depth_image(out, out_flip)
plt.imsave('onnx/DenseNet161_depth.jpg', np.log10(out), cmap='plasma_r')

end_time = time.time()  # 记录结束时间
inference_time = (end_time - start_time) * 1000  # 推断时间（ms）
print("Inference time: %.2f ms" % inference_time)
