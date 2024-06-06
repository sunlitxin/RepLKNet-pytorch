
import os
import shutil
import random
import pandas as pd

# 此代码将数据集(class:50,per_class_img:50)进行均匀随机抽取划分为25个set(per_set: 100imgs)

def create_random_image_sets(source_dir, output_dir, num_sets=25, images_per_category=2):
    # 获取所有类别目录
    categories = [d for d in os.listdir(source_dir) if os.path.isdir(os.path.join(source_dir, d))]

    # 创建类别到数值的映射字典
    category_to_number = {category: index + 1 for index, category in enumerate(categories)}

    # 确保输出目录存在
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    set_nums_dir = os.path.join(output_dir, 'set_nums')
    os.makedirs(set_nums_dir, exist_ok=True)
    all_labels = []

    # 维护每个类别的图片列表
    category_images = {category: os.listdir(os.path.join(source_dir, category)) for category in categories}

    for set_index in range(num_sets):
        set_dir = os.path.join(output_dir, f'set_{set_index + 1}')
        os.makedirs(set_dir, exist_ok=True)
        labels = []

        for category in categories:
            if len(category_images[category]) < images_per_category:
                raise ValueError(f"Not enough images in category '{category}' to create sets")

            selected_images = random.sample(category_images[category], images_per_category)
            for image in selected_images:
                category_images[category].remove(image)

                src_path = os.path.join(source_dir, category, image)
                dst_path = os.path.join(set_dir, f'{category}_{image}')
                shutil.copy(src_path, dst_path)

                # 复制到 set_nums 目录，确保文件名不添加 set 的索引前缀
                dst_nums_path = os.path.join(set_nums_dir, f'{category}_{image}')
                shutil.copy(src_path, dst_nums_path)

                labels.append([f'{category}_{image}', category, category_to_number[category]])
                all_labels.append([f'{category}_{image}', category, category_to_number[category]])

        # 打乱图片顺序
        random.shuffle(labels)

        # 写入标签文件
        labels_df = pd.DataFrame(labels, columns=['Image', 'Label', 'Label_Number'])
        labels_file = os.path.join(set_dir, 'labels.xlsx')
        labels_df.to_excel(labels_file, index=False)

    # 为 set_nums 生成标签文件
    random.shuffle(all_labels)
    all_labels_df = pd.DataFrame(all_labels, columns=['Image', 'Label', 'Label_Number'])
    labels_file = os.path.join(set_nums_dir, 'labels.xlsx')
    all_labels_df.to_excel(labels_file, index=False)


# 定义源目录和目标目录
source_dir = 'E:\data-eeg\StarImg_OnlyFace - 50\IMG_SETS\FaceEEG_IMG_test'
output_dir = 'E:\data-eeg\StarImg_OnlyFace - 50\sets_new'

# 生成随机图片集
create_random_image_sets(source_dir, output_dir)




