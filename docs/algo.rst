Installation
======================

在当前Python环境中安装Xenon-SDK的方法很简单:

.. code-block:: shell

    sudo make install_apt_deps  # 安装 apt 依赖, 需要sudo权限
    make install_pip_deps       # 安装 pip 依赖
    python setup.py install     # 安装

在镜像中构建Xenon-SDK的方法也很简单，可以参考 ``docker/README.md`` 的方法

.. code-block:: shell

    $ cd Xenon
    $ ls
      data  docs  docker dsmac  examples ...
    $ docker build . -f docker/Dockerfile -t xenon:v3.0

.. note:: 记得将 `docker/Dockerfile` 中 ``FROM`` 的镜像改为你依赖的镜像，或者你要构建的基镜像。


Make Predictions Offline
======================================

在 :ref:`Search Stage` ，  :ref:`Ensemble Stage` ，  :ref:`Predict Stage` 中，SAVEDPATH 都会保留一个 `experiment_xx_best_model.bz2` ，手动下载这个模型，
并整合到你的项目中，就可以实现离线预测（前提：Xenon-SDK已安装）

.. code-block:: python

    from joblib import load
    import pandas as pd
    xenon_workflow = load("experiment_xx_best_model.bz2")
    df = pd.read_csv("test.csv")
    y_pred = xenon_workflow.predict(df)

.. note:: `TestSet` 的数据，特征列要与 `TrainSet` 的特征列一致。

Get Column Names Before Feature Engineering
===========================================

在 :ref:`Search Stage` ，  :ref:`Ensemble Stage` ，  :ref:`Predict Stage` 中，SAVEDPATH 都会保留一个 `experiment_xx_best_model.bz2` ，手动下载这个模型，
并整合到你的项目中，然后按照以下方法可获取矢量化后进行训练的所有特征名（前提：Xenon-SDK已安装）

.. code-block:: python

    from joblib import load
    xenon_workflow = load("experiment_xx_best_model.bz2")
    # 输出矢量化后的feature_names
    print(xenon.feature_names)


Get Selected Column Names After Feature Selection
=================================================

在 :ref:`Search Stage` ，  :ref:`Ensemble Stage` 中，SAVEDPATH会额外生成 `feature_importance_xxx.csv` 和 `selected_columns_xxx.json`，分别保存的是xxx模型的特征重要度以及筛选后的特征名

其中feature_importance中的值是模型5折取平均，selected_columns中筛选保留的特征名是模型的5折进行投票，最后保留下来的。有可能会出现feature_importance会出现某个特征有值，但selected_columns最终没有存下来的情况，这是因为不同折的模型方差较大，平均下来能在feature_importance中有值但不能通过投票留存下来

selected_columns文件示例：（非真实数据仅作格式查看）

.. literalinclude:: algo/selected_columns_21598.json
   :language: json

feature_importance文件示例：（非真实数据仅作格式查看）(文件过大只截取显示一部分内容)


.. csv-table:: 
   :file: algo/feature_importance_21596.csv

External Delivery
=================================================

在 :ref:`Search Stage` ，  :ref:`Ensemble Stage` 中， 只要你设置了
 ``EXTERNAL_DELIVERY = True`` ，Xenon都会自动进行对外交付模型的构建，并保存为 `external_delivery.tar.gz`

 下载该压缩包到本地，解压，得到如下文件：

.. code-block:: bash

    .
    ├── feature_names.json
    ├── Makefile
    ├── mock_data.csv
    ├── model.bz2
    ├── test.py
    └── xenon_ext-3.0.0-py3-none-any.whl

你可以通过 ``nitrogen download 138091`` 命令获取一个
测试用的 `external_delivery.tar.gz` 

如上图所示:

+--------------------+------------------------------------------------------+
| File               | Descritprions                                        |
+====================+======================================================+
| feature_names.json | feature names                                        |
+--------------------+------------------------------------------------------+
| Makefile           |                                                      |
+--------------------+------------------------------------------------------+
| mock_data.csv      | test data, user can prepare data according to this   |
+--------------------+------------------------------------------------------+
| model.bz2          | ex-delivery model, only rely on few Xenon codes      |
+--------------------+------------------------------------------------------+
| test.py            | unit test python script                              |
+--------------------+------------------------------------------------------+
|  xenon_ext-*.whl   | ``xenon_ext`` SDK , only contain  few xenon codes    |
+--------------------+------------------------------------------------------+

- 安装 ``xenon_ext``  SDK

进入当前文件夹，首先安装 ``xenon_ext``  SDK， 命令为

.. code-block:: bash

    make install

或者


.. code-block:: bash

    pip install xenon_ext-*.whl

- 运行对外交付模型

首先介绍在当前文件夹下对对外交付模型进行单元测试。

单元测试脚本为 ``test.py`` ，除了验证对外交付模型能在当前环境下顺利跑通以外， 用户也可以参照这个脚本将对外交付模型整合到他们的系统中。

``test.py`` 逻辑为加载 ``model.bz2`` 后读取 ``mock_data.csv`` 测试数据后预测，检验对外交付模型能否顺利跑通

在终端运行：

.. code-block:: bash

    $ make test
    python test.py
    [1 1 1 1 1 1 1 1 1 1]
    # 能正常打印表示单元测试成功


如果用户需要将对外交付模型整合到他们的系统中，只需要简单的6行代码

.. code-block:: python

    import pandas as pd
    from joblib import load

    df = pd.read_csv("mock_data.csv")
    model = load("model.bz2")
    prediction = model.predict(df)
    print(prediction)



