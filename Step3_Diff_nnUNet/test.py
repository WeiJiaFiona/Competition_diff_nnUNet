import os


def list_files_in_directory(directory_path):
    try:
        # 获取指定目录下所有文件和文件夹的列表
        files_and_folders = os.listdir(directory_path)

        # 过滤出仅包含文件的列表
        files = [f for f in files_and_folders if os.path.isfile(os.path.join(directory_path, f))]

        return files
    except Exception as e:
        print(f"An error occurred: {e}")
        return []


# 示例使用
directory_path = 'D:\projects\DiffTumor\diffusion_data\GT'  # 将此处替换为实际的文件夹路径
files = list_files_in_directory(directory_path)

if files:
    print("Files in directory:")
    for file in files:
        print(file)
else:
    print("No files found or an error occurred.")
