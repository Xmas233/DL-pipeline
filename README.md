# update v0.0.1
## 更新了engine，增加了以下功能：
1. 自定义训练每n个epoch后保存一次模型，默认n=10
2. 每个epoch训练结束后释放显存

## 优化整个文件结构
新增ckpt文件夹，用于存放训练中保存的模型
model文件夹用于存放具体的模型实现
```
├─ckpt    # 存放保存模型
├─data
├─models  # 存放模型的实现
|
├─data_setup.py
├─engine.py
├─model_builder.py
├─train.py
└─utils.py

```
未来更新计划：
- [ ] 支持断点续训
- [ ] 在utils中增加绘图函数，快速查看训练样本
- [ ] 增加推理和精度评定脚本