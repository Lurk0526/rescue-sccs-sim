"""
perception/aie.py
AIE：自适应图像增强模块（Adaptive Image Enhancement）

对应论文 3.3.2节：
"在输入端嵌入自适应图像增强模块（AIE），
 针对废墟阴影区进行局部直方图均衡化拉伸"

实现：
  1. 检测图像中的阴影/暗区（废墟阴影是荧光标记漏检主因）
  2. 对暗区做 CLAHE（限制对比度自适应直方图均衡化）
  3. 亮区保持原样，避免过曝
  4. 扬尘场景额外做频域滤波去霾
"""

import cv2
import numpy as np


class AIEModule:
    """
    自适应图像增强模块
    
    CLAHE参数选择依据（论文对应）：
      clip_limit=3.0：限制对比度放大倍数，防止噪声放大
      tile_size=(8,8)：局部均衡化块大小，对应约0.5m×0.5m真实区域
    """

    def __init__(self, clip_limit=3.0, tile_size=(8, 8)):
        # CLAHE（Contrast Limited Adaptive Histogram Equalization）
        self.clahe = cv2.createCLAHE(
            clipLimit=clip_limit,
            tileGridSize=tile_size
        )
        self.shadow_thresh = 60    # 像素亮度低于此值视为阴影区
        self.dust_thresh   = 180   # 像素亮度高于此值视为扬尘高亮区

    def enhance(self, image: np.ndarray, scene_type: str = 'normal') -> np.ndarray:
        """
        主增强入口
        
        scene_type:
          'normal'   → 仅对阴影区做局部均衡化
          'dust'     → 额外做去霾（暗通道先验）
          'lowlight' → 全图伽马校正 + CLAHE
        """
        if scene_type == 'lowlight':
            return self._enhance_lowlight(image)
        elif scene_type == 'dust':
            return self._enhance_dust(image)
        else:
            return self._enhance_normal(image)

    def _enhance_normal(self, image: np.ndarray) -> np.ndarray:
        """
        正常/轻度退化：
        只对阴影区（亮度 < shadow_thresh）做 CLAHE
        亮区保持原样 → 避免荧光标记颜色失真
        """
        # 转到 LAB 色彩空间（L通道=亮度，独立处理不影响色彩）
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l_channel, a, b = cv2.split(lab)

        # 构建阴影区掩码
        shadow_mask = l_channel < self.shadow_thresh

        # 对全图做 CLAHE
        l_enhanced = self.clahe.apply(l_channel)

        # 仅在阴影区用增强结果，亮区保留原始L值
        l_out = np.where(shadow_mask, l_enhanced, l_channel)

        enhanced_lab = cv2.merge([l_out, a, b])
        return cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)

    def _enhance_lowlight(self, image: np.ndarray) -> np.ndarray:
        """
        弱光场景：
        1. 伽马校正（γ=0.5 提亮暗部）
        2. 全图 CLAHE
        """
        # 伽马校正
        gamma = 0.5
        lut = np.array([((i / 255.0) ** gamma) * 255
                        for i in range(256)], dtype=np.uint8)
        gamma_corrected = cv2.LUT(image, lut)

        # 转LAB做CLAHE
        lab = cv2.cvtColor(gamma_corrected, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        l_enhanced = self.clahe.apply(l)
        enhanced = cv2.merge([l_enhanced, a, b])
        return cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)

    def _enhance_dust(self, image: np.ndarray) -> np.ndarray:
        """
        扬尘场景：
        1. 暗通道先验去霾（Dark Channel Prior）
        2. 再做阴影区 CLAHE
        """
        dehazed = self._dark_channel_dehaze(image)
        return self._enhance_normal(dehazed)

    def _dark_channel_dehaze(self, image: np.ndarray,
                              patch_size: int = 15) -> np.ndarray:
        """
        暗通道先验去霾（简化版）
        原理：无霾图像中每个局部区域至少有一个通道值接近0
        扬尘图像全局偏亮，利用此先验估计并去除霾层
        """
        img_float = image.astype(np.float64) / 255.0

        # 计算暗通道
        dark = np.min(img_float, axis=2)
        kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT, (patch_size, patch_size))
        dark_channel = cv2.erode(dark, kernel)

        # 估计大气光（取暗通道最亮的0.1%像素对应的原图亮度）
        flat_dark = dark_channel.flatten()
        n_bright = max(1, int(len(flat_dark) * 0.001))
        bright_idx = np.argsort(flat_dark)[-n_bright:]
        atm = np.max(img_float.reshape(-1, 3)[bright_idx], axis=0)
        atm = np.clip(atm, 0.1, 1.0)

        # 估计透射率 t(x) = 1 - ω × min_c(I/A)
        omega = 0.85
        ratio = img_float / atm
        t = 1 - omega * cv2.erode(np.min(ratio, axis=2), kernel)
        t = np.clip(t, 0.15, 1.0)[:, :, np.newaxis]

        # 恢复无霾图像
        recovered = (img_float - atm) / t + atm
        recovered = np.clip(recovered * 255, 0, 255).astype(np.uint8)
        return recovered

    def detect_shadow_regions(self, image: np.ndarray) -> np.ndarray:
        """返回阴影区二值掩码（供可视化/调试用）"""
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l_channel = lab[:, :, 0]
        mask = (l_channel < self.shadow_thresh).astype(np.uint8) * 255
        return mask
