import cv2
import numpy as np
import os
from glob import glob
from tqdm import tqdm

DIR_A = ''
DIR_B = ''
DIR_OUTPUT = ''

FILE_EXTENSION = '*.png'

def ensemble_masks_bitwise():
    print(f"开始融合 (按位或)...")
    print(f"模型 A (CE):   {DIR_A}")
    print(f"模型 B (Lovasz): {DIR_B}")
    print(f"输出到: {DIR_OUTPUT}")

    os.makedirs(DIR_OUTPUT, exist_ok=True)

    search_path = os.path.join(DIR_A, FILE_EXTENSION)
    model_a_files = glob(search_path)

    if not model_a_files:
        print(f"错误：在 {DIR_A} 中没有找到任何 {FILE_EXTENSION} 文件！请检查路径和后缀。")
        return

    print(f"找到了 {len(model_a_files)} 张掩膜图，开始处理...")

    for file_path_a in tqdm(model_a_files):
        file_name = os.path.basename(file_path_a)
        file_path_b = os.path.join(DIR_B, file_name)

        if not os.path.exists(file_path_b):
            print(f"\n警告：在 {DIR_B} 中找不到对应的 {file_name}，跳过此文件。")
            continue

        mask_a = cv2.imread(file_path_a, cv2.IMREAD_GRAYSCALE)
        mask_b = cv2.imread(file_path_b, cv2.IMREAD_GRAYSCALE)

        fused_mask = cv2.bitwise_or(mask_a, mask_b)

        output_path = os.path.join(DIR_OUTPUT, file_name)
        cv2.imwrite(output_path, fused_mask)

    print(f"\n融合完成！所有 {len(model_a_files)} 张新的掩膜图已保存到：")
    print(f"{os.path.abspath(DIR_OUTPUT)}")


if __name__ == '__main__':
    ensemble_masks_bitwise()