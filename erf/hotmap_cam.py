
import torch
from torchvision.models import vgg11,resnet18,resnet101,resnext101_32x8d
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
from erf import get_model
from erf.iresnet import Self_Atten, ConcatNet_
from torchvision.models import vgg11,resnet18,resnet101,resnext101_32x8d
from pytorch_grad_cam import GradCAM,ScoreCAM,GradCAMPlusPlus,AblationCAM,XGradCAM,EigenCAM,FullGrad
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

#change 1
#prefix = 'E:/Xinyang_Github/Xinyang_ArcFace/work_dirs/ms1mv2_r100/model0.pt'

prefix = 'E:\Xinyang_Github\Xinyang_ArcFace\work_dirs\ms1mv2_r100\model0.pt'

#ke yi yun xing model r100
#prefix = 'E:\Xinyang_Github\Xinyang_ArcFace\work_dirs\ms1mv2_r100_old\model.pt'


#change 2
issame = True
#change 3  backbone
#change 4  target_layers

weight = torch.load(prefix)

if issame:
    backbone = get_model(
        'r50', dropout=0.0, fp16=True, num_features=512).cuda()

    backbone1 = Self_Atten(fp16=True, ).cuda()
    resnet = ConcatNet_(backbone, backbone1).cuda()
    # resnet.load_state_dict(weight, )
    resnet.load_state_dict(weight, strict=False)
    model = resnet
else:
    resnet = get_model(
        'r50', dropout=0.0, fp16=True, num_features=512).cuda()
    resnet.load_state_dict(weight, strict=False)
    model = resnet


# model = vgg11(pretrained=True)
#img_path = 'E:/Xinyang_Github/RepLKNet-pytorch/erf/val/Aaron_Peirsol/Aaron_Peirsol_0004.jpg'
#img_path = 'E:/Xinyang_Github/RepLKNet-pytorch/erf/val/Aaron_Patterson/Aaron_Patterson_0001.jpg'
img_path = 'E:\BaiduSyncdisk\PHD_WorkSpace\My_Paper\PR_Paper/2024MM-SA(papers)\CAM_Picture/zheng4.jpg'
#img_path = 'E:\Xinyang_Github\RepLKNet-pytorch\erf/val\Aaron_Pena\Aaron_Pena_0001.jpg'
#img_path = 'E:\Xinyang_Github\RepLKNet-pytorch\erf/val\Aaron_Sorkin\Aaron_Sorkin_0001.jpg'#2
#img_path = 'E:\Xinyang_Github\RepLKNet-pytorch\erf/val\Aaron_Tippin\Aaron_Tippin_0001.jpg'
#img_path = 'E:\Xinyang_Github\RepLKNet-pytorch\erf/val\Abbas_Kiarostami\Abbas_Kiarostami_0001.jpg'
#img_path = 'E:\Xinyang_Github\RepLKNet-pytorch\erf/val\Abba_Eban\Abba_Eban_0001.jpg'
#img_path = 'E:\Xinyang_Github\RepLKNet-pytorch\erf/val\Abdoulaye_Wade\Abdoulaye_Wade_0003.jpg'
# resize操作是为了和传入神经网络训练图片大小一致
img = Image.open(img_path).resize((112,112))
# 需要将原始图片转为np.float32格式并且在0-1之间
rgb_img = np.float32(img)/255
plt.imshow(img)
plt.show()

img_tensor=torch.from_numpy(rgb_img)
x = torch.transpose(img_tensor, 0, 2)
img_tensor = torch.unsqueeze(x, 0)
img_tensor = img_tensor.half().cuda()
# print(model)
target_layers = [model.net1.layer4[-1]]
#target_layers = [model.net2]
#target_layers = [model.layer4[-1]]
# target_layers = [model.features[-1]]#VGG
# 选取合适的类激活图，但是ScoreCAM和AblationCAM需要batch_size
cam = GradCAM(model=model,target_layers=target_layers)
targets = [ClassifierOutputTarget(201)]
print(targets)
# 上方preds需要设定，比如ImageNet有1000类，这里可以设为200
grayscale_cam = cam(input_tensor=img_tensor, targets=targets)
grayscale_cam = grayscale_cam[0, :]
cam_img = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
print(type(cam_img))
Image.fromarray(cam_img)
plt.imshow(cam_img)
plt.show()