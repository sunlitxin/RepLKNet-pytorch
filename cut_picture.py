from PIL import Image

#
# def crop_and_resize_image(image_path, new_size):
#     # 打开图像
#     image = Image.open(image_path)
#
#     # 裁剪图像
#     cropped_image = image.crop((100, 100, 100 + 369, 100 + 369))  # 替换 x、y、width、height 为你想要的裁剪区域
#
#     # 调整图像大小
#     resized_image = cropped_image.resize(new_size)
#
#     # 保存处理后的图像
#     resized_image.save("resized_image.jpg")  # 替换 "resized_image.jpg" 为你想要保存的文件名
#
#
# # 例子用法
# image_path = "E:\Xinyang_Github\RepLKNet-pytorch\Picture_ACM/0-0-0.png"  # 替换为你的图像文件路径
# new_size = (112, 112)  # 替换为你想要的新尺寸
#
# crop_and_resize_image(image_path, new_size)

#生成的热图带有白边，下面的代码可以裁切白边

# 加载图像
image = Image.open('E:\Xinyang_Github\RepLKNet-pytorch\Picture_ACM/resnet/zheng3-3.png')

# 定义要裁剪的区域
left = 144
right = 513

top = 59
bottom = 427

# 裁剪图像
cropped_image = image.crop((left, top, right, bottom))

cropped_image=cropped_image.resize((112,112))

# 保存裁剪后的图像
cropped_image.save('output.png')

