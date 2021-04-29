@[toc]
# 1. 扩展一个PyTorch编写的神经网络类的学习器


## 1.1 总结整个过程
hold on
## 1.2 在 xenon_ext 中编写模型逻辑

### 1.2.1 新建package与代码文件

首先我们需要在`xenon_ext`文件夹下把模型的逻辑写清楚

在`xenon_ext`下新建`nn`文件夹（做好包管理）
![在这里插入图片描述](https://img-blog.csdnimg.cn/2021042016105994.png)


新建`fm.py`，我们会在里面写程序逻辑



![在这里插入图片描述](https://img-blog.csdnimg.cn/20210420161242414.png)

连续coding 几个小时后，PyTorch版的FM with FTRL写好啦：

![在这里插入图片描述](https://img-blog.csdnimg.cn/20210423151631306.png)

### 1.2.2 编写基类学习器

`FMBaseEstimator` 是一个基类学习器，后续会用来派生分类器`FMClassifier`和回归器`FMRegressor`。

>以下两个表格中，**粗体**表示必选参数，*斜体*表示可选参数

在**构造函数**`__init__`中，我们会定义一些参数，如图所示。以下图代码为例子，我会说明学习器应该定义哪些参数：


|参数名| 含义 |
|--|--|
| *n_jobs* | 开启多少线程/进程 |
| *random_state* | 如果模型使用了随机算法，随机种子设置的是多少 |
| *class_weight* | 如果你的模型（分类器）是用**类别权重**来解决**类别不平衡**问题，需要定义这个参数。<br>`"balanced"`表示处理类别不平衡，`None`表示**不处理** |
| **其他参数** | 用其他参数来定义你的模型 |

在**拟合函数**`fit`中，我们会定义一些参数，如图所示。以下图代码为例子，我会说明学习器应该定义哪些参数：

|参数名| 含义 |
|--|--|
| **X** | Xenon会传一个DataFrame进来 |
| **y** | Xenon会传一个一维的np.ndarray进来 |
| *X_valid* | 如果模型不需要早停，无需设置此参数<br>验证集特征 |
| *y_valid* | 如果模型不需要早停，无需设置此参数<br>验证集标签 |
| *sample_weight* | 如果你的模型（分类器）是用**样本权重**来解决**类别不平衡**问题，需要定义这个参数。<br>长度为样本数的`一维np.ndarray`表示样本权重，`None`表示不考虑样本权重。<br>*本例中，已经在构造函数用class_weight处理了类别不平衡，所无需定义这个参数，定义出来仅为了教学目的*|


![在这里插入图片描述](https://img-blog.csdnimg.cn/20210423152422764.png)
在`FMBaseEstimator` 这个基类学习器的代码逻辑中，`is_classification`这个变量为None，可以先理解为C++的一个纯虚函数之类的，反正就是这个基类还没实现，所以先猜一个是学习器和分类器，因为不同的类型有不同的损失函数

红框中，我们通过对`is_classification`这个变量的判断来设置不同的损失函数

定义公共的预测函数`_predict`，用于被派生出的学习器和分类器调用。
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210423155912906.png)
### 1.2.3 派生子类分类器与回归器

![在这里插入图片描述](https://img-blog.csdnimg.cn/20210423194159746.png)


派生后，首先需要重载`is_classification`，才能让基类`fit`函数的逻辑能跑通

虽然基类实现了`fit`函数，但因为预测逻辑不一样，所以分类器与回归器分别实现`predict`函数。需要注意的是，分类器需要额外多实现一个`predict_proba`函数，返回的是每个类别的。

## 1.3 编写对xenon_ext模型的单元测试

编写单元测试`tests/ext_models/test_fm.py`，能够顺利跑通预设任务，测试成功

![在这里插入图片描述](https://img-blog.csdnimg.cn/20210424094649834.png)

## 1.4 将模型注册进xenon workflow系统中

### 1.4.1 注册分类器FMClassifier

新建`xenon/workflow/components/classification/pytorch_fm.py`

![在这里插入图片描述](https://img-blog.csdnimg.cn/20210424095210162.png)

Xenon 的 workflow系统采用**代理模式**来**注册**新的学习器






```python
from xenon.workflow.components.classification_base import XenonClassificationAlgorithm

__all__ = ["FMClassifier"]


class FMClassifier(XenonClassificationAlgorithm):
    class__ = "FMClassifier"
    module__ = "xenon_ext.nn"

    def core_fit(self, estimator, X, y, X_valid=None, y_valid=None, X_test=None,
                 y_test=None, feature_groups=None):
        estimator.fit(X, y, X_valid, y_valid)
        return estimator
```

>以下表格中，**粗体**表示必选参数，*斜体*表示可选参数


>**注意**，`__all__ = ["FMClassifier"]`必填，workflow系统 需要根据这个变量来识别类


|需要overwrite的 参数/方法| 含义 |
|--|--|
| **class__** | 被注册学习器的类名 |
| **module__** | 被注册学习器的module路径 |
| *core_fit* | 如果模型的`fit`函数只介绍`X,y`参数，即`fit(X,y)`，无需重载该方法<br>如果模型的`fit`函数参数复杂，如本例中的`fit(X,y,X_valid,y_valid)`，需要重载|

### 1.4.2 注册回归器FMRegressor

同理，创建`xenon/workflow/components/regression/pytorch_fm.py`文件，并写以下内容：

```python
from xenon.workflow.components.regression_base import XenonRegressionAlgorithm

__all__ = ["FMRegressor"]


class FMRegressor(XenonRegressionAlgorithm):
    class__ = "FMRegressor"
    module__ = "xenon_ext.nn"

    def core_fit(self, estimator, X, y, X_valid=None, y_valid=None, X_test=None,
                 y_test=None, feature_groups=None):
        estimator.fit(X, y, X_valid, y_valid)
        return estimator
```


## 1.5 对注册后的模型进行单元测试

新建文件`tests/algorithm_component/test_fm.py`

因为注册后的学习器接受`DataFrameContainer`和`NdArrayContainer`参数，所以我们对参数进行处理后，传入注册后的学习器：


```python
import os

from sklearn import datasets
from sklearn.model_selection import train_test_split

from xenon.data_container import DataFrameContainer
from xenon.data_container import NdArrayContainer
from xenon.tests.base import LocalResourceTestCase
from xenon.workflow.components.classification.pytorch_fm import FMClassifier
from xenon.workflow.components.regression.pytorch_fm import FMRegressor


class TestFM(LocalResourceTestCase):
    def setUp(self) -> None:
        super(TestFM, self).setUp()
        self.plot_dir = os.getcwd() + "/test_iter_algorithm"
        from pathlib import Path
        Path(self.plot_dir).mkdir(parents=True, exist_ok=True)

    def test_classifier(self):
        X, y = datasets.load_digits(return_X_y=True)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                                                            random_state=0)
        X_train = DataFrameContainer("TrainSet", dataset_instance=X_train, resource_manager=self.mock_resource_manager)
        X_test = DataFrameContainer("TestSet", dataset_instance=X_test, resource_manager=self.mock_resource_manager)
        y_train = NdArrayContainer("TrainLabel", dataset_instance=y_train, resource_manager=self.mock_resource_manager)
        y_test = NdArrayContainer("TestLabel", dataset_instance=y_test, resource_manager=self.mock_resource_manager)
        FMClassifier().fit(X_train, y_train, X_test, y_test)

    def test_regressor(self):
        X, y = datasets.load_boston(return_X_y=True)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                                                            random_state=0)
        X_train = DataFrameContainer("TrainSet", dataset_instance=X_train, resource_manager=self.mock_resource_manager)
        X_test = DataFrameContainer("TestSet", dataset_instance=X_test, resource_manager=self.mock_resource_manager)
        y_train = NdArrayContainer("TrainLabel", dataset_instance=y_train, resource_manager=self.mock_resource_manager)
        y_test = NdArrayContainer("TestLabel", dataset_instance=y_test, resource_manager=self.mock_resource_manager)
        FMRegressor().fit(X_train, y_train, X_test, y_test)
```

单元测试可顺利跑通

![在这里插入图片描述](https://img-blog.csdnimg.cn/20210424102441469.png)

## 1.6 在 hdl_bank.json 中定义学习器的超参空间

xenon采用超参描述语言HDL定义超参空间，相关介绍在xenon_opt的文档中，这里不赘述

根据FM构造函数的参数，我们对这些参数的取值范围进行设置：


- 红色部分参数是必须要指定的，具体含义见1.2.2的说明
- 粉红色参数为一些常量，这是是我根据经验写死的
- 蓝色为一些条件参数。`k`（FM隐藏大小）是在`use_fm=true`时才有效的（`use_fm=false`时退化为逻辑回归）

![在这里插入图片描述](https://img-blog.csdnimg.cn/20210424105731946.png)

```json
    "pytorch_fm": {
      "n_jobs": 1,
      "random_state": 42,
      // 虽然实现了sgd和adam优化器，但是用ftrl求解会得到泛化能力更好的模型
      "optimizer": "ftrl",
      "tol": 1e-3,
      "max_epoch": 1000,
      "k": {"_type": "int_quniform", "_value": [1, 4, 1],"_default": 2},
      "use_fm": {"_type": "choice", "_value": [true, false],"_default": true},
      "alpha": {"_type": "uniform", "_value": [0.1, 2],"_default": 1.0},
      "beta": {"_type": "uniform", "_value": [0.1, 2],"_default": 1.0},
      "l2": {"_type": "uniform", "_value": [0.1, 2],"_default": 1.0},
      "l1": {"_type": "uniform", "_value": [0.1, 2],"_default": 1.0},
      "__condition":[
          {
            "_child": "k",
            "_parent": "use_fm",
            "_values": [true]
          },
        ]
    },
```
## 1.7 如有必要，更新search时的默认学习器候选集

`scripts/env_configs/search.json:80`

![在这里插入图片描述](https://img-blog.csdnimg.cn/20210424110635983.png)
## 1.8 在nitrogen上进行线上测试


```json
{
    "name": "test_fm",
    "description": "",
    "git_url": "git@bitbucket.org:xtalpi/xenon.git",
    "git_branch": "extend_models",
    "git_commit": "",
    "datasets": "170692",
    "command": "python scripts/search.py",
    "use_gpu": 0,
    "docker_image": "477093822308.dkr.ecr.us-east-2.amazonaws.com/nitrogen-1/xenon:v2.1",
    "project": "X-P012-V2.0",
    "cluster": "aws",
    "resource": {
        "cpu": 48,
        "memory": 192,
        "spot": true
    },
    "env": {
        "USER_ID": "2",
        "USER_TOKEN": "8v$NdlCVujOey#&194fK%7OwYc8FNsMY",
        "TRAIN_TARGET_COLUMN_NAME": "label",
        "SPECIFIC_TASK_TOKEN": "test_fm",
        "RANDOM_RUNS": "40",
        "EXTERNAL_DELIVERY": "False",
        "SEARCH_THREAD_NUM": "6",
        "N_ITERATIONS": "50",
        "ENSEMBLE_SIZE": "3",
        "DISPLAY_SIZE": "10",
        "TOTAL_TIME_LIMIT": "144000",
        "USE_BOHB": "False",
        "CLASSIFIER": "[\"pytorch_fm\"]",
        "OPT_EARLY_STOP_ROUNDS": "512",
        "DATASET_NAME": "hERG_1-1",
        "MODEL_TYPE": "clf"
    }
}
```



![在这里插入图片描述](https://img-blog.csdnimg.cn/20210424132102772.png)顺利跑通，结果还可


# 2. 扩展Scaler

## 2.1 修改 flexible_scaler.py

修改文件：	`xenon_ext/scale/flexible_scaler.py`

如图，只需要做少量的修改，就可以扩展`RobustScaler`
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210426084948198.png)

## 2.2 如有必要，更新search时的默认学习器候选集

修改文件：	`scripts/env_configs/search.json`

如图，新增了策略名
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210426085145249.png)

# 3 扩展Selector

## 3.1 修改 flexible_feature_selector.py

修改文件：`xenon_ext/feature_selection/flexible_feature_selector.py`

目前整合了这几种特征筛选策略：

- gbdt，梯度替身树，由lightgbm实现
- rf，随机森林，可选 random_forest 或 extra_trees
- l1_linear，L1 正则化的线性模型，通过产生稀疏解来做特征筛选
- none，不做特征筛选


参数如下：

![在这里插入图片描述](https://img-blog.csdnimg.cn/2021041911534493.png)

| 变量名 | 变量类型 | 取值范围|
|--|--|--|
| strategy | choice | ["none", "l1_linear", "rf", "gbdt"] |
| should_select_percent | choice |[true, false] |
| select_percent | quniform | [10,100,0.5] |
| rf_type | choice | ["random_forest", "extra_trees"] |
| C | loguniform | [0.01, 10] |

假设我们一个简单的扩展：**引入xgboost做特征筛选**，我们可以这样修改：




在xenon代码中使用命令`git diff 4d1279c2d81acbea94619ded438d9589dff505ca cad32f2f9d87ff5c2bc5fa756f77036b7eecf829`
可以看到：

![在这里插入图片描述](https://img-blog.csdnimg.cn/20210426090357193.png)
## 3.2 如有必要，更新search时的默认学习器候选集

修改文件：	`scripts/env_configs/search.json`

如图，新增了策略名


![在这里插入图片描述](https://img-blog.csdnimg.cn/20210426090948701.png)

