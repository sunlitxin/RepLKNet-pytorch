
import os

#该代码实现了对当前目录下的所有子文件夹中的图片进行重新命名

def rename_images_in_subdirectories(parent_directory):
    # 遍历主目录中的所有子目录
    for root, dirs, files in os.walk(parent_directory):
        for directory in dirs:
            # 构建子目录的完整路径
            subdirectory = os.path.join(root, directory)
            # 调用函数重命名子目录中的图片
            rename_images(subdirectory)


def rename_images(directory):
    # 获取指定文件夹中的所有文件
    files = os.listdir(directory)

    # 过滤出图片文件（假设是.jpg和.png格式）
    image_files = [file for file in files if file.endswith(('.jpg', '.jpeg', '.png'))]

    # 按原始文件名排序，以确保重命名的一致性
    image_files.sort()

    # 创建一个临时的重命名路径，以避免命名冲突
    temp_directory = os.path.join(directory, "temp_rename")
    os.makedirs(temp_directory, exist_ok=True)

    # 遍历图片文件并重命名到临时目录
    for i, filename in enumerate(image_files):
        # 获取文件扩展名
        ext = os.path.splitext(filename)[1]
        # 新的文件名
        new_name = f"{i}{ext}"
        # 构建完整的文件路径
        old_path = os.path.join(directory, filename)
        temp_path = os.path.join(temp_directory, new_name)
        # 移动文件到临时目录
        os.rename(old_path, temp_path)
        print(f"Temporarily renamed {old_path} to {temp_path}")

    # 将临时目录中的文件移回原始目录
    temp_files = os.listdir(temp_directory)
    temp_files.sort()

    for filename in temp_files:
        temp_path = os.path.join(temp_directory, filename)
        new_path = os.path.join(directory, filename)
        # 移动文件回原始目录
        os.rename(temp_path, new_path)
        print(f"Renamed {temp_path} to {new_path}")

    # 删除临时目录
    os.rmdir(temp_directory)


# 指定上级文件夹路径
parent_directory = r"E:\data-eeg\StarImg_OnlyFace - 50\CelebritiesFaces\Male_Celebrities"

# 调用函数重命名所有子目录中的图片
rename_images_in_subdirectories(parent_directory)
