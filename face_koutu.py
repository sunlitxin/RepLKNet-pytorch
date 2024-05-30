import os
import cv2

#下面代码实现了对人脸图像的定位并将人脸图像裁切下来

face_count = 0
def detect_and_save_faces(image_path, output_dir,face_count):
    # 加载人脸识别分类器
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # 读取图像
    image = cv2.imread(image_path)

    # 将图像转换为灰度图像
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 检测人脸
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # 在图像上绘制人脸边界框并保存人脸区域
    for (x, y, w, h) in faces:
        # 增加矩形框的尺寸
        padding = 50  # 你可以调整这个值以增加或减少边距
        x_new = max(0, x - padding)
        y_new = max(0, y - padding)
        w_new = min(image.shape[1] - x_new, w + 2 * padding)
        h_new = min(image.shape[0] - y_new, h + 2 * padding)

        # 截取人脸区域
        face_region = image[y_new:y_new + h_new, x_new:x_new + w_new]

        # 保存人脸区域
        face_filename = os.path.join(output_dir, f'{face_count}.jpg')

        cv2.imwrite(face_filename, face_region)


# 文件路径和保存目录
input_dir = 'E:\CV_Data\Face-EEG\StarImg\chineseface\\songdandan'
output_dir = 'E:\data-eeg\StartImg_OnlyFace\\songdandan'

# 确保保存目录存在
if not os.path.exists(output_dir):
    os.makedirs(output_dir)


# 遍历图片目录中的所有文件
for filename in os.listdir(input_dir):
    #--------------防止中文字符报错先修改img名字--------------
    # 获取文件的扩展名
    _, ext = os.path.splitext(filename)
    # 构建新的文件名
    new_filename = f"{face_count}{ext}"
    # 重命名文件
    os.rename(os.path.join(input_dir, filename), os.path.join(input_dir, new_filename))
    #----------------------------------------------------

    #face_count++
    face_count += 1

    # 构造完整的文件路径
    image_path = os.path.join(input_dir, filename)

    # 检查文件是否是图片
    if os.path.isfile(image_path) and any(image_path.endswith(ext) for ext in ['.jpg', '.jpeg', '.png', '.bmp']):
        # 调用函数进行人脸检测和保存
        print(image_path)
        detect_and_save_faces(image_path, output_dir, face_count)
