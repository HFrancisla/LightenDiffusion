import os
import torch
import torch.utils.data
from PIL import Image
from datasets.data_augment import PairCompose, PairToTensor, PairRandomHorizontalFilp, PairRandomCrop


class SICEDataset:
    """
    SICE 配对数据集包装类，用于第一阶段训练。
    SICE 数据集提供同一场景不同曝光级别的配对图像，
    这对反射率一致性损失至关重要。
    """
    def __init__(self, config):
        self.config = config

    def get_loaders(self):
        train_dataset = SICEPairDataset(
            self.config.data.data_dir,
            patch_size=self.config.data.patch_size,
            filelist='sice_train.txt',
            train=True
        )
        val_dataset = SICEPairDataset(
            self.config.data.data_dir,
            patch_size=self.config.data.patch_size,
            filelist='sice_val.txt',
            train=False
        )

        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=self.config.training.batch_size,
            shuffle=True,
            num_workers=self.config.data.num_workers,
            pin_memory=True,
            drop_last=True
        )
        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=self.config.data.num_workers,
            pin_memory=True
        )

        return train_loader, val_loader


class SICEPairDataset(torch.utils.data.Dataset):
    """
    SICE 配对数据集：每个样本包含同一场景的两张不同曝光图像
    (I^1_low, I^2_low)，如论文 Section 3.4 所述。

    文件列表格式（每行）：
        img1_path img2_path
    其中 img1 和 img2 是同一场景的不同曝光图像。
    """
    def __init__(self, data_dir, patch_size, filelist='sice_train.txt', train=True):
        super().__init__()

        self.data_dir = data_dir
        self.patch_size = patch_size
        self.train = train

        list_path = os.path.join(data_dir, filelist)
        with open(list_path) as f:
            contents = f.readlines()
            self.pairs = [line.strip() for line in contents if line.strip()]

        if train:
            self.transforms = PairCompose([
                PairRandomCrop(patch_size),
                PairRandomHorizontalFilp(),
                PairToTensor()
            ])
        else:
            self.transforms = PairCompose([
                PairToTensor()
            ])

    def __getitem__(self, index):
        pair_line = self.pairs[index]
        img1_path, img2_path = pair_line.split(' ')[0], pair_line.split(' ')[1]

        img1 = Image.open(img1_path).convert('RGB')
        img2 = Image.open(img2_path).convert('RGB')

        # 确保两张图像尺寸一致（如需要，将 img2 缩放到 img1 尺寸）
        if img1.size != img2.size:
            img2 = img2.resize(img1.size, Image.BICUBIC)

        # 训练时如需要进行填充（裁剪要求最小尺寸）
        if self.train:
            w, h = img1.size
            if w < self.patch_size or h < self.patch_size:
                pad_w = max(self.patch_size - w, 0)
                pad_h = max(self.patch_size - h, 0)
                import torchvision.transforms.functional as F
                img1 = F.pad(img1, (0, 0, pad_w, pad_h), fill=0, padding_mode='reflect')
                img2 = F.pad(img2, (0, 0, pad_w, pad_h), fill=0, padding_mode='reflect')

        img1, img2 = self.transforms(img1, img2)

        img_id = os.path.basename(img1_path)

        # 返回两张图像拼接为 6 通道：[img1(3通道) | img2(3通道)]
        return torch.cat([img1, img2], dim=0), img_id

    def __len__(self):
        return len(self.pairs)
