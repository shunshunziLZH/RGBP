任务：实现并运行最小小样本过拟合测试，不修改模型结构。

要求：
1. 从训练集中固定选取 5 张样本。
2. 关闭随机翻转、随机裁剪、颜色扰动等随机增强。
3. 保持 model(rgb, x) 接口不变。
4. 使用当前 restoration head 输出 pred。
5. 使用 L1Loss(pred, target)。
6. 训练 500~1000 iterations。
7. 每 50 iterations 打印 loss。
8. 每 100 iterations 保存一次可视化结果：
   - rgb input
   - polar_x 的前 3 个通道
   - pred
   - target
9. 不新增 SSIM/perceptual/physical loss。
10. 不修改 CM-FRM、FFM、backbone。