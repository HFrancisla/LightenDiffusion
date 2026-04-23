"""
LightenDiffusion 第一阶段训练脚本。

使用 SICE 数据集的配对多曝光图像训练编码器 (feature_pyramid + channel_down)、
CTDN (Retinex_decom) 和解码器 (ReconNet 解码分支)。

扩散模型 (UNet) 不参与本阶段。

损失函数 (论文 Section 3.4):
    L_con  = || I - D(E(I)) ||_2              (内容重建损失)
    L_rec  = || F - R_i * L_j ||_1            (Retinex 重建损失)
    L_ref  = || R_1 - R_2 ||_1                (反射率一致性损失)
    L_ill  = || ∇L * exp(-λ_g * ∇R) ||_2     (光照平滑损失)
    L_dec  = L_rec + λ_ref * L_ref + λ_ill * L_ill

用法:
    python train_stage1.py --config stage1.yml
"""

import argparse
import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import yaml

import datasets
import utils
from models.decom import CTDN


def parse_args_and_config():
    parser = argparse.ArgumentParser(description="LightenDiffusion Stage 1 Training")
    parser.add_argument(
        "--config", default="stage1.yml", type=str, help="Path to the config file"
    )
    parser.add_argument(
        "--resume", default="", type=str, help="Path for checkpoint to load and resume"
    )
    parser.add_argument(
        "--seed",
        default=230,
        type=int,
        help="Seed for initializing training (default: 230)",
    )
    args = parser.parse_args()

    with open(os.path.join("configs", args.config), "r") as f:
        config = yaml.safe_load(f)
    new_config = dict2namespace(config)

    return args, new_config


def dict2namespace(config):
    namespace = argparse.Namespace()
    for key, value in config.items():
        if isinstance(value, dict):
            new_value = dict2namespace(value)
        else:
            new_value = value
        setattr(namespace, key, new_value)
    return namespace


class Stage1Trainer:
    """
    第一阶段训练器：使用 Retinex 分解损失优化编码器 + CTDN + 解码器。
    """

    def __init__(self, config):
        self.config = config
        self.device = config.device

        # 初始化 CTDN（包含 ReconNet + Retinex_decom）
        self.model = CTDN(channels=64)
        self.model.to(self.device)
        self.model = torch.nn.DataParallel(self.model)

        # 损失函数
        self.l1_loss = nn.L1Loss()
        self.l2_loss = nn.MSELoss()

        # 优化器（论文：Adam，lr=1e-4，衰减系数 0.8）
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=config.optim.lr,
            weight_decay=config.optim.weight_decay,
            betas=(0.9, 0.999),
            amsgrad=config.optim.amsgrad,
            eps=config.optim.eps,
        )

        # 学习率调度器
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer,
            step_size=config.training.lr_decay_every,
            gamma=config.training.lr_decay_factor,
        )

        self.step = 0

    def load_checkpoint(self, path):
        checkpoint = utils.logging.load_checkpoint(path, None)
        self.model.load_state_dict(checkpoint["state_dict"], strict=True)
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.step = checkpoint["step"]
        if "scheduler" in checkpoint:
            self.scheduler.load_state_dict(checkpoint["scheduler"])
        else:
            # 兼容旧检查点：手动推进 scheduler 到正确位置
            for _ in range(self.step):
                self.scheduler.step()
        print("=> loaded checkpoint '{}' (step {}, lr={:.6f})".format(
            path, self.step, self.optimizer.param_groups[0]["lr"]))

    def save_checkpoint(self, filename):
        """保存检查点，兼容第二阶段加载格式。"""
        # 保存完整检查点，用于恢复训练
        utils.logging.save_checkpoint(
            {
                "step": self.step,
                "state_dict": self.model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "scheduler": self.scheduler.state_dict(),
            },
            filename=os.path.join(self.config.data.ckpt_dir, filename),
        )

        # 保存仅模型权重的检查点，格式兼容第二阶段加载：
        #   第二阶段通过 model.load_state_dict(checkpoint['model'], strict=True) 加载
        #   其中 model 是裸 CTDN()（非 DataParallel 包装）
        if isinstance(self.model, nn.DataParallel):
            model_state = self.model.module.state_dict()
        else:
            model_state = self.model.state_dict()

        torch.save(
            {"model": model_state},
            os.path.join(self.config.data.ckpt_dir, "stage1_weight.pth.tar"),
        )

    def compute_gradient(self, x):
        """计算水平和垂直方向梯度。"""
        grad_h = x[:, :, :, :-1] - x[:, :, :, 1:]
        grad_v = x[:, :, :-1, :] - x[:, :, 1:, :]
        return grad_h, grad_v

    def illumination_smooth_loss(self, L, R, lambda_g):
        """
        光照平滑损失（论文 Eq.9）：
        L_ill = || ∇L · exp(-λ_g · ∇R) ||_2

        鼓励光照图局部平滑，同时在反射率边缘处允许不连续。
        """
        L_grad_h, L_grad_v = self.compute_gradient(L)
        R_grad_h, R_grad_v = self.compute_gradient(R)

        smooth_h = torch.abs(L_grad_h) * torch.exp(-lambda_g * torch.abs(R_grad_h))
        smooth_v = torch.abs(L_grad_v) * torch.exp(-lambda_g * torch.abs(R_grad_v))

        loss = torch.mean(smooth_h**2) + torch.mean(smooth_v**2)
        return loss

    def compute_losses(self, img1, img2):
        """
        计算一对不同曝光图像的所有第一阶段损失。

        参数:
            img1: 第一张曝光图像 [B, 3, H, W]
            img2: 第二张曝光图像 [B, 3, H, W]

        返回:
            loss_dict: 包含所有损失值的字典

        显存分析:
            feature_pyramid 在 512×512 下每次调用保存 ~7 个 [B,64,512,512] 级
            激活张量用于反向传播，每次约 B×750MB。本方法共执行 3 次 pyramid 调用：
              - 第 1 次前向 (pred_fea=None): pyramid(img1) + pyramid(img2) = 2 次
              - 第 2 次前向 (pred_fea=fea1): pyramid(img1) 用于解码跳跃连接 = 1 次
            总计 ~3 × B × 750MB 激活。batch=4 时约 9GB，加上模型/优化器 ~11GB。
        """
        lambda_ref = self.config.loss.lambda_ref
        lambda_ill = self.config.loss.lambda_ill
        lambda_g = self.config.loss.lambda_g

        images_cat = torch.cat([img1, img2], dim=1)  # [B, 6, H, W]

        # ==================== 前向传播：编码 + 分解 ====================
        # 单次前向传播获取潜在特征和 Retinex 分解结果
        output = self.model(images_cat, pred_fea=None)
        fea1 = output["low_fea"]  # E(img1): 潜在特征 [B, 3, H/8, W/8]
        fea2 = output["high_fea"]  # E(img2): 潜在特征 [B, 3, H/8, W/8]
        R1, L1 = output["low_R"], output["low_L"]
        R2, L2 = output["high_R"], output["high_L"]

        # ==================== 内容重建损失 (Eq.7) ====================
        # L_con = || I - D(E(I)) ||_2
        #
        # 论文只给出公式 ||I - D(E(I))||_2，D(E(·)) 的具体数据流根据
        # ReconNet 代码 (decom.py L161-186) 推断：
        #   1. 将 pred_fea (3ch) 通过 channel_up 还原为 256ch 高维特征
        #   2. 重新运行 pyramid(I) 获取 down2/down4/down8 的跳跃连接特征
        #   3. 将 channel_up(pred_fea) 与跳跃连接逐级相加并上采样
        #
        # 注意：由于解码器经由跳跃连接可直接获取原图的多尺度特征，
        # L_con 对瓶颈层 (3ch latent) 的约束可能较弱；分解质量主要
        # 依赖 L_rec、L_ref、L_ill 三个分解损失。但此解码路径与第二阶段
        # 推理时一致，必须沿用以保持架构兼容。
        #
        # 显存优化：只对 img1 计算 L_con（省去 1 次 pyramid 前向传播）。
        # feature_pyramid 在 512×512 全分辨率下运行，每次调用约占
        # batch×750MB 的激活显存。编码器/解码器权重共享，单图梯度
        # 已能充分训练重建能力。
        #
        # CTDN.forward 在 pred_fea 非空时只使用 images[:, :3] 提供跳跃
        # 连接，后 3 通道被忽略，此处传入任意 6ch 即可。
        recon1 = self.model(
            torch.cat([img1, img1], dim=1), pred_fea=fea1
        )["pred_img"]
        loss_con = self.l2_loss(recon1, img1)

        # ==================== 分解损失 ====================
        # Retinex 重建损失 (Eq.8):
        # L_rec = ΣΣ || F_j - R_i ⊙ L_j ||_1
        loss_rec = (
            self.l1_loss(fea1, R1 * L1)
            + self.l1_loss(fea2, R1 * L2)
            + self.l1_loss(fea1, R2 * L1)
            + self.l1_loss(fea2, R2 * L2)
        )

        # 反射率一致性损失 (Eq.9):
        # L_ref = || R_1 - R_2 ||_1
        loss_ref = self.l1_loss(R1, R2)

        # 光照平滑损失 (Eq.9):
        # L_ill = Σ || ∇L_i · exp(-λ_g · ∇R_i) ||_2
        loss_ill = self.illumination_smooth_loss(
            L1, R1, lambda_g
        ) + self.illumination_smooth_loss(L2, R2, lambda_g)

        # 总分解损失
        loss_dec = loss_rec + lambda_ref * loss_ref + lambda_ill * loss_ill

        # 第一阶段总损失
        total_loss = loss_con + loss_dec

        return {
            "total": total_loss,
            "con": loss_con,
            "rec": loss_rec,
            "ref": loss_ref,
            "ill": loss_ill,
            "dec": loss_dec,
        }

    @torch.no_grad()
    def validate(self, val_loader):
        """在验证集上评估分解质量。"""
        self.model.eval()
        r_consistency_sum, l_std_sum = 0., 0.
        n = 0

        for x, _ in val_loader:
            x = x.to(self.device)
            img1, img2 = x[:, :3, ...], x[:, 3:, ...]

            images_cat = torch.cat([img1, img2], dim=1)
            output = self.model(images_cat, pred_fea=None)
            R1, R2 = output["low_R"], output["high_R"]
            L1, L2 = output["low_L"], output["high_L"]

            r_consistency_sum += torch.mean(torch.abs(R1 - R2)).item()
            l_std_sum += (torch.std(L1).item() + torch.std(L2).item()) / 2.0
            n += 1

        if n > 0:
            print("[Val step {}] R_consistency: {:.6f}, L_std: {:.6f}".format(
                self.step, r_consistency_sum / n, l_std_sum / n))

        self.model.train()

    def train(self, dataset):
        cudnn.benchmark = True
        train_loader, val_loader = dataset.get_loaders()

        n_iters = self.config.training.n_iters
        print("Starting Stage 1 training for {} iterations...".format(n_iters))

        self.model.train()
        epoch = 0

        while self.step < n_iters:
            epoch += 1
            data_start = time.time()

            for i, (x, img_id) in enumerate(train_loader):
                if self.step >= n_iters:
                    break

                self.step += 1
                x = x.to(self.device)

                # 拆分为两张不同曝光的图像
                img1 = x[:, :3, ...]
                img2 = x[:, 3:, ...]

                # 计算所有损失
                losses = self.compute_losses(img1, img2)
                total_loss = losses["total"]

                self.optimizer.zero_grad()
                total_loss.backward()
                self.optimizer.step()
                self.scheduler.step()

                # 日志输出
                if self.step % 10 == 0:
                    elapsed = time.time() - data_start
                    print(
                        "step:{}/{}, loss:{:.5f} (con:{:.5f} rec:{:.5f} ref:{:.5f} ill:{:.5f}) "
                        "lr:{:.6f} time:{:.3f}s".format(
                            self.step,
                            n_iters,
                            total_loss.item(),
                            losses["con"].item(),
                            losses["rec"].item(),
                            losses["ref"].item(),
                            losses["ill"].item(),
                            self.optimizer.param_groups[0]["lr"],
                            elapsed,
                        )
                    )
                    data_start = time.time()

                # 验证
                if self.step % self.config.training.validation_freq == 0:
                    self.validate(val_loader)

                # 保存检查点
                if self.step % self.config.training.save_freq == 0:
                    self.save_checkpoint("model_step_{}".format(self.step))
                    print("=> saved checkpoint at step {}".format(self.step))

            print("Epoch {} finished (step {}/{})".format(epoch, self.step, n_iters))

        # 最终保存
        self.save_checkpoint("model_final")
        print(
            "Stage 1 training completed. Weights saved to {}/stage1_weight.pth.tar".format(
                self.config.data.ckpt_dir
            )
        )


def main():
    args, config = parse_args_and_config()

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print("Using device: {}".format(device))
    config.device = device

    # 设置随机种子
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = True

    # 创建检查点目录
    os.makedirs(config.data.ckpt_dir, exist_ok=True)

    # 加载数据集
    print("=> using dataset '{}'".format(config.data.train_dataset))
    DATASET = datasets.__dict__[config.data.type](config)

    # 创建训练器并开始训练
    trainer = Stage1Trainer(config)

    if os.path.isfile(args.resume):
        trainer.load_checkpoint(args.resume)

    trainer.train(DATASET)


if __name__ == "__main__":
    main()
