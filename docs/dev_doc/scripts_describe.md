# 代码文件组织
|文件夹名| 功能 |
|--|--|
| docker | 存放各个版本镜像构建时的Dockerfile |
| docs | Xenon文档 |
| dsmac | 在smac基础上改的，通过数据库同步的方法实现了多机并行。原代码在[automl/SMAC3](https://github.com/automl/SMAC3) |
| generic_fs | 原本想写一个与操作系统无关的文件系统，后用来存放一些http、io方面的代码 |
| scripts | 一些在nitrogen、renova上跑的脚本，通过获取环境变量的方式来获取用户设置的参数 |
| tests | 单元测试脚本 |
| xenon | 核心代码 |
| xenon_ext | xenon扩展，用于放一些我们自己开发的学习器和特征工程。之所以将学习器与特征工程代码放这，是为了方便对外交付 |
| xenon_opt | xenon优化器，为了方便调试所以将代码放这，完整的xenon_opt项目见对应的仓库 |

`xenon/ ` 核心代码文件组织

|文件夹名| 功能 |
|--|--|
| core | 核心区，主要在`xenon/core/base.py`实现了`XenonEstimator` ，并派生了分类器和回归器|
| data_container | 数据容器，在xenon的工作流中，X是`DataFrameContainer`，y是`NdArrayContainer`|
| ensemble | 存放集成学习相关的代码|
| estimator | 历史遗留代码，已迁移到xenon_ext，忽略|
| evaluation | 实现了评价器，主要功能是提供一个目标函数，这个函数将传入的config进行实例化为机器学习管线，然后在训练集上训练，验证集上打分，并返回loss|
| hdl | 超参描述语言相关的功能，主要用于构造机器学习的参数空间 |
| interpret | 可解释性相关的代码，目前实现了特征重要度 |
| metrics | 评价指标 |
| resource_manager | 资源管理器，做数据库连接、后端连接这种事情 |
| tests | 提供一些用于单元测试的基类 |
| tools | 提供一些工具，目前实现了对外交付功能 |
| utils | 所有零碎的支撑性功能都在这个文件夹下实现 |
| workflow | 机器学习工作流系统，如分类器、回归器、特征工程 |