import cv2
import numpy as np
from PIL import Image, ImageEnhance
from skimage import exposure, restoration
import tensorflow as tf


def enhance_image(image_path):
    # 读取图像
    image = cv2.imread(image_path)

    # 去噪
    denoised_image = cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 21)

    # 锐化
    kernel = np.array([[0, -1, 0],
                       [-1, 5, -1],
                       [0, -1, 0]])
    sharpened_image = cv2.filter2D(src=denoised_image, ddepth=-1, kernel=kernel)

    # # 转换为PIL格式以增强对比度
    # pil_image = Image.fromarray(cv2.cvtColor(sharpened_image, cv2.COLOR_BGR2RGB))
    # enhancer = ImageEnhance.Contrast(pil_image)
    # enhanced_image = enhancer.enhance(1.5)
    #
    # # 再次转换为OpenCV格式以进行颜色校正
    # final_image = cv2.cvtColor(np.array(enhanced_image), cv2.COLOR_RGB2BGR)
    # gamma_corrected_image = exposure.adjust_gamma(final_image, 1.2)

    # 保存最终增强后的图像
    cv2.imwrite('E:\Xinyang_Github\RepLKNet-pytorch\erf\\val\Aaron_Guiel\\final_enhanced_image.jpg', sharpened_image)


# 调用函数
enhance_image('E:\Xinyang_Github\RepLKNet-pytorch\erf\\val\Aaron_Peirsol\Aaron_Peirsol_0001.jpg')
