Installation
----------------------

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
--------------------------------------------

在 :ref:`Search Stage` ，  :ref:`Ensemble Stage` ，  :ref:`Predict Stage` 中，SAVEDPATH 都会保留一个 `experiment_xx_best_model.bz2` ，手动下载这个模型，
并整合到你的项目中，就可以实现离线预测（前提：Xenon-SDK已安装）

.. code-block:: python

    from joblib import load
    import pandas as pd
    xenon_workflow = load("experiment_xx_best_model.bz2")
    df = pd.read_csv("test.csv")
    y_pred = xenon_workflow.predict(df)

.. note:: `TestSet` 的数据，特征列要与 `TrainSet` 的特征列一致。

