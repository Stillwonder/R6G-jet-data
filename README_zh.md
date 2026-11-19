# 射流论文数据与代码整理包


- `data/per_frame.csv`：3027帧，含32工况的标签、面积分数、单侧最大跨度、两类计数、Ba、Bl、PC1及原图哈希。
- `data/per_condition.csv`：32工况的均值、样本数、S0--S5组成、标准差与论文中已有的bootstrap区间。
- `metadata/pca_parameters.json`：完整精度的PCA载荷、log变换均值和样本标准差、协方差和解释方差比。
- `scripts/`：统计重算、bootstrap、PCA核对、mask提取与图3、7、8、9绘图脚本。
- `examples/raw/`：S0--S5各一张，直接复制原始JPEG，保留原始像素和文件哈希。`examples/representatives.csv`给出与统计表的对应行。
- `examples/masks/`、`examples/previews/`：可复现的外部区域mask及明确标识的派生预览，不是“原图”。
- `figures/reference/`：V38原图PDF。`figures/reproduced/`：用本包脚本实际生成的图。
- `README.md`：完整英文说明，含全部列名定义、单位、条件编码、随机种子、图像参数、联系人和许可范围。

进入文件夹后，可依次运行：

```bash
python -m pip install -r requirements.txt
python scripts/reproduce_statistics.py
python scripts/extract_image_features.py
python scripts/make_mask_previews.py
python scripts/plot_figures.py
python scripts/verify_repository.py
```

脚本默认相对于自身所在仓库查找输入，不依赖本机原路径；可用`--root`和`--output-dir`指定新位置。只有在另有完整原图目录时，才用`extract_image_features.py --raw-root ...`重提取3027帧。本包自带六张例图，不声称六张原图可以替代全部原始数据。

3027帧的S0--S5状态均为人工标注。跨度ell本身已是单侧跨度，不应再除以二；图3的141个点是从原矢量图恢复的实验点位，不是仪器原始导出。Ba/Bl为空表示按原规则不适用，不能按零处理。

数据、文档和作者图像拟按CC BY 4.0，代码按MIT，许可证文件已分开放置。公开存储前仍需作者确认共同作者许可和数据/图片权利，DOI为：10.5281/zenodo.24359532
