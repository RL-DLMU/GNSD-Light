# 自适应交通信号控制
本项目使用多智能体强化学习解决自适应交通信号控制中的不同路口具有不同重要性的现实问题。以下内容介绍了项目的环境要求以及使用方式。

## 环境要求：
- 操作系统：Windows/Linux/MacOS（已在Windows 以及Linux 上测试）。
- Python版本：推荐使用Python 3.7。
- 主要依赖包括：
  - PyTorch
  - NumPy
  - Matplotlib
  - Math
## 使用方式：
## 训练模型
- 直接运行 `run.py` 即可
- `python GNSD-Light/run.py`

本项目为论文 [Sequential Decision MARL for Adaptive Traffic Signal Control With Different Intersections Priorities](https://ieeexplore.ieee.org/document/11021536) 的官方开源实现，如有引用请使用以下 BibTeX：
```bibtex
@ARTICLE{11021536,
  author={Liu, Wanting and Zhang, Chengwei and Zhou, Kailing and Li, Yihong and Zhan, Furui and Xue, Wanli and Chen, Rong},
  journal={IEEE Transactions on Intelligent Transportation Systems}, 
  title={Sequential Decision MARL for Adaptive Traffic Signal Control With Different Intersections Priorities}, 
  year={2025},
  volume={26},
  number={7},
  pages={9799-9811}}
