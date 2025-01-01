import os


def rename_images(input_dir):
    """
    将文件夹中的图片文件名从 image_xxxx.png 改为 mask_xxxx.png。

    Args:
        input_dir (str): 图片文件所在的文件夹路径。
    """
    if not os.path.exists(input_dir):
        print(f"目录 {input_dir} 不存在。")
        return

    for filename in os.listdir(input_dir):
        if filename.startswith("image_") and filename.endswith(".png"):
            # 提取 xxxx 部分
            suffix = filename[6:]  # 去掉 "image_" 的前缀
            new_filename = f"mask_{suffix}"
            old_filepath = os.path.join(input_dir, filename)
            new_filepath = os.path.join(input_dir, new_filename)

            # 重命名文件
            os.rename(old_filepath, new_filepath)
            print(f"已将 {filename} 重命名为 {new_filename}")

    print("重命名完成。")


# 使用示例
input_directory = "/workspace/psen/SplattingAvatar-master/dataset/people/customize/spl_male/masks"  # 替换为你的图片文件夹路径
rename_images(input_directory)
