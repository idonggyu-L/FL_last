# -*- coding = utf-8 -*-
import cv2
import torch
import numpy as np
from skimage import io
from skimage.util import img_as_ubyte
from torchvision.transforms import transforms


def add_trigger(args, image, test=False):
    pixel_max = max(1,torch.max(image))
    if args.attack in ('dba',) and (test in (False, True)):
        # 4분할 트리거: 조각 s×s를 간격 g로 2×2 배치(전체 폭 2s+g).
        # g>0 이면 조각들이 떨어져 각자 독립 국소패턴(정통 DBA). g=0 이면 통짜 블록(구버전).
        s = int(getattr(args, 'dba_size', 2))
        g = int(getattr(args, 'dba_gap', 0))
        ys = [args.triggerY, args.triggerY + s + g]
        xs = [args.triggerX, args.triggerX + s + g]
        if test:                                  # 평가: 4조각 전부(간격 유지) 합체
            cells = [(yi, xi) for yi in (0, 1) for xi in (0, 1)]
        else:                                     # 학습: 자기 조각 하나만
            c = int(args.dba_class)
            cells = [(c // 2, c % 2)]
        for yi, xi in cells:
            image[:, ys[yi]:ys[yi] + s, xs[xi]:xs[xi] + s] = pixel_max
        if not test:
            args.save_img(image)
        return image
    if args.trigger == 'square':
        pixel_max = torch.max(image) if torch.max(image) > 1 else 1

        if args.dataset == 'cifar' or  args.dataset == 'cifar100':
            pixel_max = 1
        image[:, args.triggerY:args.triggerY + 5, args.triggerX:args.triggerX + 5] = pixel_max
        # image[:, 25:32, 25:32] = 1
    elif args.trigger == 'square3':
        pixel_max = torch.max(image) if torch.max(image) > 1 else 1

        if args.dataset == 'cifar' or  args.dataset == 'cifar100':
            pixel_max = 1
        image[:, args.triggerY:args.triggerY + 3, args.triggerX:args.triggerX + 3] = pixel_max
    elif args.trigger == 'square7':
        pixel_max = torch.max(image) if torch.max(image) > 1 else 1

        if args.dataset == 'cifar' or  args.dataset == 'cifar100':
            pixel_max = 1
        image[:, args.triggerY-2:args.triggerY + 5, args.triggerX-2:args.triggerX + 5] = pixel_max

    elif args.trigger == 'square10':
        pixel_max = torch.max(image) if torch.max(image) > 1 else 1

        if args.dataset == 'cifar' or  args.dataset == 'cifar100':
            pixel_max = 1
        image[:, args.triggerY-5:args.triggerY + 5, args.triggerX-5:args.triggerX + 5] = pixel_max
    elif args.trigger == 'pattern':
        pixel_max = torch.max(image) if torch.max(image) > 1 else 1
        image[:, args.triggerY + 0, args.triggerX + 0] = pixel_max
        image[:, args.triggerY + 1, args.triggerX + 1] = pixel_max
        image[:, args.triggerY - 1, args.triggerX + 1] = pixel_max
        image[:, args.triggerY + 1, args.triggerX - 1] = pixel_max
    elif args.trigger == 'watermark':
        if args.watermark is None:
            args.watermark = cv2.imread('./utils/watermark.png', cv2.IMREAD_GRAYSCALE)
            args.watermark = cv2.bitwise_not(args.watermark)
            args.watermark = cv2.resize(args.watermark, dsize=image[0].shape, interpolation=cv2.INTER_CUBIC)
            pixel_max = np.max(args.watermark)
            args.watermark = args.watermark.astype(np.float64) / pixel_max
            # cifar [0,1] else max>1
            pixel_max_dataset = torch.max(image).item() if torch.max(image).item() > 1 else 1
            args.watermark *= pixel_max_dataset
        max_pixel = max(np.max(args.watermark), torch.max(image))
        watermark = torch.Tensor(args.watermark).to(image.device)
        image += watermark
        image[image > max_pixel] = max_pixel

    elif args.trigger == 'apple':
        if args.apple is None:
            args.apple = cv2.imread('./utils/apple.png', cv2.IMREAD_GRAYSCALE)
            args.apple = cv2.bitwise_not(args.apple)
            args.apple = cv2.resize(args.apple, dsize=image[0].shape, interpolation=cv2.INTER_CUBIC)
            pixel_max = np.max(args.apple)
            args.apple = args.apple.astype(np.float64) / pixel_max
            # cifar [0,1] else max>1
            pixel_max_dataset = torch.max(image).item() if torch.max(image).item() > 1 else 1
            args.apple *= pixel_max_dataset
        max_pixel = max(np.max(args.apple), torch.max(image))
        # print(image,args.apple)
        apple = torch.Tensor(args.apple).to(image.device)
        image += apple
        image[image > max_pixel] = max_pixel

        # args.save_img(image)
    elif args.trigger == 'blend':
        # if args.hallokitty is None:
        #     args.hallokitty = cv2.imread('./utils/halloKitty.png')
        #     print(args.hallokitty.shape)
        #     pixel_max = np.max(args.hallokitty)
        #     args.hallokitty = args.hallokitty.astype(np.float64) / pixel_max
        #     args.hallokitty = torch.from_numpy(args.hallokitty)
        #     # cifar [0,1] else max>1
        #     pixel_max_dataset = torch.max(image).item() if torch.max(image).item() > 1 else 1
        #     args.hallokitty *= pixel_max_dataset
        # print(args.hallokitty.shape)
        # # hallokitty = torch.Tensor(args.hallokitty).to(image.device)
        # hallokitty = args.hallokitty.float().to(image.device)  # 显式转为 float32
        # print(hallokitty.shape, image.shape)
        # image = hallokitty * 0.2 + image * 0.8
        # max_pixel = max(torch.max(args.hallokitty), torch.max(image))
        # image[image > max_pixel] = max_pixel
        # print('save')
        trigger = HelloKittyTrigger('./utils/halloKitty.png', alpha=0.2, args=args, device=args.device)
        image = trigger(image)

        # save_tensor_as_image(image,'./figs/hallokitty.png')
        # save_poisoned_images(image, )
    elif args.trigger == 'sig':
        trigger = SigTriggerAttack(args=args, device=args.device)
        image = trigger(image)
    elif args.trigger == 'LFT':
        trigger = LowFrequencyTrigger(args=args, device=args.device)
        image = trigger(image)
    # save the most recent backdoor image in test dataset
    # args.save_img(image)
    return image

import torchvision.utils as vutils
import os
def save_tensor_as_image(tensor, filename):
    # tensor = tensor.cpu().permute(1, 2, 0).numpy()  # [C,H,W] → [H,W,C]
    # tensor = (tensor * 255).astype('uint8')  # [0,1] → [0,255]]
    tensor = (tensor + 1) / 2.0

    # 2. 处理batch维度（如果是4D Tensor）
    if len(tensor.shape) == 4:
        tensor = tensor[0]  # 取第一张图片 [C,H,W]

    # 3. 转为NumPy并调整维度
    tensor = tensor.cpu().permute(1, 2, 0).numpy()  # [C,H,W] → [H,W,C]

    # 4. 转换为0-255范围
    tensor = (tensor * 255).astype('uint8')
    cv2.imwrite(filename, cv2.cvtColor(tensor, cv2.COLOR_RGB2BGR))  # OpenCV需要BGR


def save_mnist_tensor_as_image(tensor, filename):
    # 确保 tensor 是 CPU 上的张量
    tensor = tensor.detach().cpu()

    # 处理batch维度（如果是3D Tensor）
    if len(tensor.shape) == 3:
        tensor = tensor[0]  # 取第一张图片 [1,H,W]

    # 转为NumPy并调整维度
    tensor = tensor.squeeze(0).numpy()  # [1,H,W] → [H,W]

    # 转换为0-255范围
    tensor = (tensor * 255).astype('uint8')

    # 保存图像
    cv2.imwrite(filename, tensor)
class HelloKittyTrigger:
    def __init__(self, trigger_path, alpha=1, args=None, device='cuda'):
        """
        Args:
            trigger_path: Hello Kitty图片路径
            alpha: 混合比例（0.2表示触发器占比20%）
            device: 触发器所在的设备（需与输入图像一致）
        """
        # 1. 读取触发器并预处理为Tensor [C, H, W]
        if args.blend is not None:
            trigger = args.blend  # 直接使用blend
        else:
            trigger = cv2.imread(trigger_path)  # [H, W, C] BGR格式
            trigger = cv2.resize(trigger, (32, 32))  # 调整为CIFAR-10尺寸
            # save_tensor_as_image(trigger, './figs/hallokitty_trigger_resize.png')
            trigger = cv2.cvtColor(trigger, cv2.COLOR_BGR2RGB)  # 转RGB
            # save_tensor_as_image(trigger, './figs/hallokitty_trigger_rgb.png')
            trigger = torch.from_numpy(trigger).float() / 255.0  # [0,1]范围
            trigger = trigger.permute(2, 0, 1)  # [H, W, C] → [C, H, W]

        # 2. 归一化触发器（匹配输入图像的归一化方式，假设输入已归一化）
        self.trigger = trigger.cpu()  # 移动到指定设备
        # save_tensor_as_image(self.trigger, './figs/hallokitty_trigger.png')
        # print('max:',torch.max(self.trigger),torch.min(self.trigger))
        self.alpha = alpha
        self.args = args

    def __call__(self, img_tensor):
        """
        Args:
            img_tensor: 输入图像 [C, H, W] 或 [B, C, H, W]，范围假设为[0,1]或[-1,1]
        Returns:
            混合后的图像（与输入同范围）
        """
        # 3. 混合触发器（自动支持有无batch维度）
        # print(self.trigger,img_tensor)
        img_tensor.cpu()

        blended = (1 - self.alpha) * img_tensor.cpu() + self.alpha * self.trigger
        # save_tensor_as_image(blended, './figs/hallokitty_blend.png')
        # save_tensor_as_image(img_tensor.cpu(), './figs/hallokitty_origin.png')

        # 4. 确保数值范围合法（根据输入范围调整）
        max_pixel = max(torch.max(self.trigger), torch.max(img_tensor))
        blended[blended > max_pixel] = max_pixel
        # if torch.min(img_tensor) >= -1:  # 假设输入范围可能是[0,1]或[-1,1]
        #     blended = torch.clamp(blended, 0, 1)  # 限制到[0,1]
        # else:
        #     blended = torch.clamp(blended, -1, 1)  # 限制到[-1,1]
        # save_tensor_as_image(blended.cpu(), './figs/hallokitty_clip.png')

        return blended

import  math
class SigTriggerAttack:
    """
    Sinusoidal backdoor trigger (PyTorch Tensor版本)
    Args:
        delta: 振幅（默认40）
        f: 频率（默认6）
        img_size: 图像尺寸（如(32,32)）
    """

    def __init__(self,
                 delta=40,
                 f=6,
                 img_size=(32, 32),
                 device='cuda',
                 args=None,
                 alpha=0.2):
        self.delta = delta
        self.f = f
        self.img_size = img_size
        self.device = device
        self.alpha = alpha

        # 预生成正弦波pattern [C,H,W]
        if args.sig is  not None:
            self.pattern = args.sig
        else:
            self.pattern = self._generate_pattern()
            args.sig = self.pattern

    def _generate_pattern(self):
        """生成正弦波pattern"""
        h, w = self.img_size
        j = torch.arange(w, device=self.device).float()
        raw_pattern = self.delta * torch.sin(2 * math.pi * j * self.f / w)  # [W]

        # 2. 归一化到[-1,1]（假设输入图像已归一化）
        # 公式：pattern = (raw_pattern / max_abs_value) * trigger_strength
        max_abs_value = torch.max(torch.abs(raw_pattern))
        normalized_pattern = (raw_pattern / max_abs_value)  # [-1,1]

        # 3. 扩展到 [C,H,W]
        normalized_pattern = normalized_pattern.repeat(h, 1)  # [H,W]
        normalized_pattern = normalized_pattern.unsqueeze(0).repeat(3, 1, 1)  # [C,H,W]
        save_tensor_as_image(normalized_pattern, './figs/sig_pattern.png')
        return normalized_pattern

    def __call__(self, img_tensor):
        """
        Args:
            img_tensor: 输入图像 [C,H,W] 或 [B,C,H,W]，范围[-1,1]（假设已归一化）
        Returns:
            带触发器的图像（同输入范围）
        """
        if len(img_tensor.shape) == 4:  # Batch输入 [B,C,H,W]
            pattern = self.pattern.unsqueeze(0)
        else:
            pattern = self.pattern

            # 直接叠加（无需反归一化）
        img_poisoned = (1 - self.alpha) * img_tensor.cpu() + self.alpha * pattern.cpu()
        save_tensor_as_image(img_tensor,'./figs/sig_origin.png')
        save_tensor_as_image(img_poisoned, './figs/sig.png')
        max_pixel = max(torch.max(self.pattern.cpu()), torch.max(img_tensor.cpu()))
        img_poisoned[img_poisoned > max_pixel] = max_pixel
        return img_poisoned


import torch
import numpy as np
import logging


class LowFrequencyTrigger:
    """
    低频触发器（PyTorch Tensor 版本）
    Args:
        trigger_path: 低频模式.npy文件路径
        img_size: 目标图像尺寸 (H, W)
        device: 设备 ('cuda' 或 'cpu')
    """

    def __init__(self, trigger_path, img_size=(32, 32), device='cuda', args=None, alpha=0.2):
        # 加载并预处理触发器
        trigger_array = np.load(trigger_path)

        # 处理不同维度的输入
        if len(trigger_array.shape) == 4:
            logging.info("低频触发器为4维，取第一个")
            trigger_array = trigger_array[0]
        elif len(trigger_array.shape) == 3:
            pass
        elif len(trigger_array.shape) == 2:
            trigger_array = np.stack((trigger_array,) * 3, axis=-1)
        else:
            raise ValueError("低频触发器维度错误，应为2/3/4维")

        # 转换为PyTorch Tensor并归一化到[-1, 1]
        self.args = args
        if args.LFT is not None:
            self.trigger = args.LFT
        else:
            self.trigger = self._preprocess_trigger(trigger_array, img_size, device)
            args.LFT = self.trigger
        logging.info(f"低频触发器加载完成，形状: {self.trigger.shape}")

    def _preprocess_trigger(self, trigger_np, img_size, device):
        """预处理触发器"""
        # 调整尺寸
        trigger_resized = cv2.resize(trigger_np, (img_size[1], img_size[0]))

        # 转换为Tensor并归一化到[-1, 1]
        trigger = torch.from_numpy(trigger_resized).float()
        if trigger.max() > 1:  # 假设原始范围[0,255]
            trigger = (trigger / 127.5) - 1  # [0,255] -> [-1,1]

        # 调整维度顺序 [H,W,C] -> [C,H,W]
        trigger = trigger.permute(2, 0, 1).to(device)
        save_tensor_as_image(trigger, './figs/LFT_pattern.png')

        return trigger

    def __call__(self, img_tensor):
        """
        Args:
            img_tensor: 输入图像 [C,H,W] 或 [B,C,H,W]，范围[-1,1]
        Returns:
            带触发器的图像（范围仍为[-1,1]）
        """
        if len(img_tensor.shape) == 4:  # Batch输入
            trigger = self.trigger.unsqueeze(0)
        else:
            trigger = self.trigger

        # 直接叠加触发器
        img_poisoned = img_tensor + trigger
        save_tensor_as_image(img_poisoned, './figs/LFT_img.png')


        # 裁剪到[-1,1]范围
        max_pixel = max(torch.max(trigger.cpu()), torch.max(img_tensor.cpu()))
        img_poisoned[img_poisoned > max_pixel] = max_pixel
        save_tensor_as_image(img_poisoned, './figs/LFT.png')

        return img_poisoned


class SSBAAttack:
    """
    SSBA 攻击（PyTorch Tensor 版本）
    Args:
        replace_images_path: 替换图像.npy文件路径
        img_size: 目标图像尺寸 (H, W)
        device: 设备 ('cuda' 或 'cpu')
    """

    def __init__(self, replace_images_path, img_size=(32, 32), device='cuda', args=None):
        # 加载替换图像集
        self.replace_images = np.load(replace_images_path)
        logging.info(f"加载SSBA替换图像，形状: {self.replace_images.shape}")
        self.args =args

        # 预处理的替换图像（Tensor格式）
        if args.ssbs is not None:
            self.preprocessed_images = args.ssbs
        else:
            self.preprocessed_images = self._preprocess_images(self.replace_images, img_size, device)
            args.ssbs = self.preprocessed_images
        self.device = device

    def _preprocess_images(self, images_np, img_size, device):
        """预处理替换图像集"""
        processed = []
        for img in images_np:
            # 调整尺寸
            img_resized = cv2.resize(img, (img_size[1], img_size[0]))
            # 转换为Tensor并归一化
            img_tensor = torch.from_numpy(img_resized).float()
            if img_tensor.max() > 1:  # [0,255] -> [0,1]
                img_tensor = img_tensor / 255.0
            # [H,W,C] -> [C,H,W]
            img_tensor = img_tensor.permute(2, 0, 1)
            processed.append(img_tensor)

        # 堆叠为Tensor [N,C,H,W]
        return torch.stack(processed).to(device)

    def __call__(self, img_tensor, image_serial_id=None):
        """
        Args:
            img_tensor: 输入图像（实际不使用）
            image_serial_id: 用于选择替换图像的索引
        Returns:
            替换后的图像 Tensor [C,H,W]
        """
        if image_serial_id is None:
            raise ValueError("SSBA攻击需要提供image_serial_id")

        return self.preprocessed_images[image_serial_id]


# 使用示例


if __name__ == "__main__":
# 使用示例
    trigger = HelloKittyTrigger('../data/hello_kitty.jpeg', alpha=0.2, device='cuda')

    # 在训练循环中直接使用
    for x, y in train_loader:
        x = x.to('cuda')  # 输入图像已是Tensor
        x_poisoned = trigger(x)  # 添加触发器
        # 继续训练...