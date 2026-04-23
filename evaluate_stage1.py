"""
LightenDiffusion 第一阶段验证脚本。

可视化训练完成的第一阶段模型的分解结果（反射率图和光照图），用于验证分解质量。

关键验证标准：
    1. 反射率图 (R) 应包含丰富内容，且在不同曝光下保持一致
    2. 光照图 (L) 应无内容结构（平滑，无场景结构）
    3. 重建图像 D(E(I)) 应与输入高度一致

用法:
    python evaluate_stage1.py --input_dir /path/to/test_images --ckpt ckpt/stage1/stage1_weight.pth.tar
    python evaluate_stage1.py --input_pair img1.png img2.png --ckpt ckpt/stage1/stage1_weight.pth.tar
"""

import argparse
import os
import glob
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import torchvision.transforms.functional as TF
import torchvision.utils as tvu

from models.decom import CTDN


def load_model(ckpt_path, device):
    """加载训练完成的第一阶段 CTDN 模型。"""
    model = CTDN(channels=64)
    checkpoint = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(checkpoint['model'], strict=True)
    model.to(device)
    model.eval()
    print("=> Loaded Stage 1 weights from '{}'".format(ckpt_path))
    return model


def load_image(path, device):
    """加载单张图像并转换为张量 [1, 3, H, W]。"""
    img = Image.open(path).convert('RGB')
    tensor = TF.to_tensor(img).unsqueeze(0).to(device)
    return tensor


def pad_to_multiple(tensor, multiple=64):
    """将张量空间尺寸填充为 `multiple` 的整数倍。"""
    _, _, h, w = tensor.shape
    pad_h = (multiple - h % multiple) % multiple
    pad_w = (multiple - w % multiple) % multiple
    if pad_h > 0 or pad_w > 0:
        tensor = nn.functional.pad(tensor, (0, pad_w, 0, pad_h), mode='reflect')
    return tensor, h, w


def save_tensor_as_image(tensor, path):
    """将 [1, C, H, W] 张量保存为图像文件。"""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tvu.save_image(tensor.clamp(0, 1), path)


def visualize_single(model, img_path, output_dir, device):
    """
    对单张图像进行分解并保存：
      - 输入图像
      - 编码后的潜在特征（上采样可视化）
      - 反射率图 R
      - 光照图 L
      - 重建图像 D(E(I))
    """
    img = load_image(img_path, device)
    img_padded, orig_h, orig_w = pad_to_multiple(img, multiple=8)
    img_name = os.path.splitext(os.path.basename(img_path))[0]

    with torch.no_grad():
        # 编码：需要 6 通道输入，复制图像
        images_6ch = torch.cat([img_padded, img_padded], dim=1)
        output = model(images_6ch, pred_fea=None)

        fea = output["low_fea"]      # [1, 3, H/8, W/8]
        R = output["low_R"]          # [1, 3, H/8, W/8]
        L = output["low_L"]          # [1, 3, H/8, W/8]

        # 重建：D(E(I))
        recon = model(images_6ch, pred_fea=fea)["pred_img"]
        recon = recon[:, :, :orig_h, :orig_w]

        # 上采样 R、L、fea 用于可视化（它们在 1/8 分辨率）
        R_vis = nn.functional.interpolate(R, size=(orig_h, orig_w), mode='bilinear', align_corners=False)
        L_vis = nn.functional.interpolate(L, size=(orig_h, orig_w), mode='bilinear', align_corners=False)
        fea_vis = nn.functional.interpolate(fea, size=(orig_h, orig_w), mode='bilinear', align_corners=False)

    img = img[:, :, :orig_h, :orig_w]

    # 保存各项结果
    save_dir = os.path.join(output_dir, img_name)
    save_tensor_as_image(img, os.path.join(save_dir, 'input.png'))
    save_tensor_as_image(fea_vis, os.path.join(save_dir, 'latent_feature.png'))
    save_tensor_as_image(R_vis, os.path.join(save_dir, 'reflectance_R.png'))
    save_tensor_as_image(L_vis, os.path.join(save_dir, 'illumination_L.png'))
    save_tensor_as_image(recon, os.path.join(save_dir, 'reconstructed.png'))

    # 保存合并对比图
    grid = tvu.make_grid(
        [img[0], fea_vis[0], R_vis[0], L_vis[0], recon[0]],
        nrow=5, padding=4, pad_value=1.0
    )
    tvu.save_image(grid, os.path.join(save_dir, 'comparison.png'))

    print("  [{}] saved to {}".format(img_name, save_dir))


def visualize_pair(model, img1_path, img2_path, output_dir, device):
    """
    对一对不同曝光的图像进行分解并保存：
      - 两张输入图像
      - 反射率图 R1、R2（应看起来相似）
      - 光照图 L1、L2（应无内容结构）
      - 反射率差异图 |R1 - R2|（应接近零）
    """
    img1 = load_image(img1_path, device)
    img2 = load_image(img2_path, device)

    # 如需要，将 img2 缩放到与 img1 相同尺寸
    if img1.shape != img2.shape:
        img2 = nn.functional.interpolate(img2, size=img1.shape[2:], mode='bilinear', align_corners=False)

    img1_padded, orig_h, orig_w = pad_to_multiple(img1, multiple=8)
    img2_padded, _, _ = pad_to_multiple(img2, multiple=8)

    pair_name = "{}__vs__{}".format(
        os.path.splitext(os.path.basename(img1_path))[0],
        os.path.splitext(os.path.basename(img2_path))[0]
    )

    with torch.no_grad():
        images_6ch = torch.cat([img1_padded, img2_padded], dim=1)
        output = model(images_6ch, pred_fea=None)

        R1, L1 = output["low_R"], output["low_L"]
        R2, L2 = output["high_R"], output["high_L"]

        # 上采样用于可视化
        R1_vis = nn.functional.interpolate(R1, size=(orig_h, orig_w), mode='bilinear', align_corners=False)
        R2_vis = nn.functional.interpolate(R2, size=(orig_h, orig_w), mode='bilinear', align_corners=False)
        L1_vis = nn.functional.interpolate(L1, size=(orig_h, orig_w), mode='bilinear', align_corners=False)
        L2_vis = nn.functional.interpolate(L2, size=(orig_h, orig_w), mode='bilinear', align_corners=False)

        # 反射率差异（放大以便观察）
        R_diff = torch.abs(R1_vis - R2_vis)
        R_diff_amplified = (R_diff * 5.0).clamp(0, 1)

        # 计算定量指标
        r_consistency = torch.mean(torch.abs(R1 - R2)).item()
        l1_content_score = torch.std(L1).item()
        l2_content_score = torch.std(L2).item()

    img1 = img1[:, :, :orig_h, :orig_w]
    img2 = img2[:, :, :orig_h, :orig_w]

    save_dir = os.path.join(output_dir, pair_name)

    # 保存各项结果
    save_tensor_as_image(img1, os.path.join(save_dir, 'input_1.png'))
    save_tensor_as_image(img2, os.path.join(save_dir, 'input_2.png'))
    save_tensor_as_image(R1_vis, os.path.join(save_dir, 'reflectance_R1.png'))
    save_tensor_as_image(R2_vis, os.path.join(save_dir, 'reflectance_R2.png'))
    save_tensor_as_image(L1_vis, os.path.join(save_dir, 'illumination_L1.png'))
    save_tensor_as_image(L2_vis, os.path.join(save_dir, 'illumination_L2.png'))
    save_tensor_as_image(R_diff_amplified, os.path.join(save_dir, 'reflectance_diff_5x.png'))

    # 保存合并对比网格
    # 第1行：input1, R1, L1
    # 第2行：input2, R2, L2
    # 第3行：R差异（放大）
    grid = tvu.make_grid(
        [img1[0], R1_vis[0], L1_vis[0],
         img2[0], R2_vis[0], L2_vis[0],
         R_diff_amplified[0], R_diff_amplified[0], R_diff_amplified[0]],
        nrow=3, padding=4, pad_value=1.0
    )
    tvu.save_image(grid, os.path.join(save_dir, 'comparison_pair.png'))

    print("  [{}]".format(pair_name))
    print("    Reflectance consistency (L1, lower=better): {:.6f}".format(r_consistency))
    print("    Illumination std (L1, lower=more uniform):  {:.6f}".format(l1_content_score))
    print("    Illumination std (L2, lower=more uniform):  {:.6f}".format(l2_content_score))
    print("    Results saved to {}".format(save_dir))

    return r_consistency, l1_content_score, l2_content_score


def main():
    parser = argparse.ArgumentParser(description='LightenDiffusion Stage 1 Evaluation')
    parser.add_argument('--ckpt', default='ckpt/stage1/stage1_weight.pth.tar', type=str,
                        help='Path to Stage 1 checkpoint')
    parser.add_argument('--input_dir', default='', type=str,
                        help='Directory containing test images (decompose each individually)')
    parser.add_argument('--input_pair', nargs=2, default=None, type=str,
                        help='Two image paths for paired decomposition comparison')
    parser.add_argument('--pair_list', default='', type=str,
                        help='Text file with paired images (same format as sice_train.txt)')
    parser.add_argument('--output_dir', default='results/stage1_eval', type=str,
                        help='Output directory for visualization results')
    args = parser.parse_args()

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print("Using device: {}".format(device))

    model = load_model(args.ckpt, device)

    # 模式 1：单张图像分解
    if args.input_dir:
        print("\n=== Single Image Decomposition ===")
        img_paths = sorted(
            glob.glob(os.path.join(args.input_dir, '*.png')) +
            glob.glob(os.path.join(args.input_dir, '*.jpg')) +
            glob.glob(os.path.join(args.input_dir, '*.bmp'))
        )
        if not img_paths:
            print("No images found in '{}'".format(args.input_dir))
        for path in img_paths:
            visualize_single(model, path, os.path.join(args.output_dir, 'single'), device)

    # 模式 2：配对分解对比
    if args.input_pair:
        print("\n=== Paired Decomposition Comparison ===")
        visualize_pair(model, args.input_pair[0], args.input_pair[1],
                       os.path.join(args.output_dir, 'pair'), device)

    # 模式 3：从文件列表批量配对验证
    if args.pair_list and os.path.isfile(args.pair_list):
        print("\n=== Batch Paired Evaluation ===")
        with open(args.pair_list) as f:
            pairs = [line.strip() for line in f if line.strip()]

        all_r_cons, all_l1_std, all_l2_std = [], [], []
        for line in pairs:
            parts = line.split(' ')
            if len(parts) >= 2:
                r_con, l1_std, l2_std = visualize_pair(
                    model, parts[0], parts[1],
                    os.path.join(args.output_dir, 'pair_batch'), device
                )
                all_r_cons.append(r_con)
                all_l1_std.append(l1_std)
                all_l2_std.append(l2_std)

        if all_r_cons:
            print("\n=== Summary ({} pairs) ===".format(len(all_r_cons)))
            print("  Avg reflectance consistency: {:.6f}".format(np.mean(all_r_cons)))
            print("  Avg illumination std:        {:.6f}".format(np.mean(all_l1_std + all_l2_std)))

    if not args.input_dir and not args.input_pair and not args.pair_list:
        print("Please specify at least one of: --input_dir, --input_pair, --pair_list")
        print("Example:")
        print("  python evaluate_stage1.py --input_dir test_images/ --ckpt ckpt/stage1/stage1_weight.pth.tar")
        print("  python evaluate_stage1.py --input_pair img1.png img2.png --ckpt ckpt/stage1/stage1_weight.pth.tar")
        print("  python evaluate_stage1.py --pair_list sice_val.txt --ckpt ckpt/stage1/stage1_weight.pth.tar")


if __name__ == "__main__":
    main()
