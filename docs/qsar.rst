
QSAR Usage Overview
==================================

`QSAR(Quantitative structure–activity relationship)` 是指定量的构效关系，是使用数学模型来描述分子结构和分子的某种生物活性之间的关系。其基本假设是化合物的分子结构包含了决定其物理，化学及生物等方面的性质信息，而这些理化性质则进一步决定了该化合物的生物活性。进而，化合物的分子结构性质数据与其生物活性也应该存在某种程度上的相关。



``Xenon`` 是XARC自动化机器学习平台，在 `QSAR` 问题中， ``Xenon`` 的作用域是:
    1. 建立一个在训练集上以某种验证方法（如5折交叉）和评价指标（如r2）下表现最好的模型，可以是集成学习的模型。
    2. 可以用这个模型对未知数据进行预测

.. image:: https://gitee.com/TQCAI/xenon_iamge/raw/master/13.png
    :width: 600px

在Nitrogen上使用 ``Xenon`` 各阶段输入的数据类型如下表所示：



.. csv-table:: Data Structure of Stages
   :file: data_structure_of_stages.csv

名词解释:
    1. ``Dataset`` 指的是Nitrogen Job的数据集
    2. ``Options`` 指的是选项，填在 `Command` 中，填写格式为 ``--option=xxx`` 。如 **display步骤** 的 ``--task_id={task_id}``
    3. ``Argument`` 指的是参数，填在 `Command` 中，填写格式为 ``arg1 arg2 ...`` 。如 **ensemble步骤** 的 ``213 214``
    4. ``ENV`` 指的是环境变量。所有的环境变量都在 :ref:`ENV Description Table` 中有详细说明

-----------------------------------------------------------------------------------

在Nitrogen上使用 ``Xenon`` 的整体流程如图所示：

.. image:: https://gitee.com/TQCAI/xenon_iamge/raw/master/11.png

Usage Process In Nitrogen
==================================

Register and Login
----------------------------------

``Xenon`` 的权限管理采用的是公司的XACS模块，所以使用 ``Xenon`` 前，你需要注册一个XACS账号。你可以进入 `CosMos <https://x-cosmos.net/#/login?redirect=%2F>`_ 网站注册XACS账号，注册后，系统会发送一封激活邮件到你的工作邮箱，确认后你的账户就可用了。

完成注册后，你需要登录 ``Xenon`` 。首先你需要安装 ``xenon_cli`` 。有两种安装方法:
    1. clone xenon 仓库后从源码安装
    2. 下载安装包后用pip安装

1. **clone xenon 仓库后从源码安装**

.. code-block:: bash

   $ git clone -b v3.0 git@bitbucket.org:xtalpi/xenon.git
   $ cd xenon_client
   $ python setup.py install

2. **下载安装包后用pip安装**

.. note:: ``xenon_cli`` 需要 `Python3.6` 以上的Python环境。

.. image:: https://gitee.com/TQCAI/xenon_iamge/raw/master/9.png
    :width: 600px

Vectorization Stage
----------------------------------

:download:`Download example script for Vectorization-Stage <nitrogen_example_temp/vectorization.json>`.

.. literalinclude:: nitrogen_example_temp/vectorization.json
   :language: json

首先我们需要提供一个 ``训练数据csv`` ，

这个csv一定要有这3个列:
    1. ``NAME`` , 主键
    2. ``SMILES`` , 用于矢量化
    3. ``pIC50`` / ``active`` , 训练目标

.. note:: ``训练数据csv`` 前两列必须是 ``["NAME", "SMILES"]`` , 否则矢量化时会报错。

完成矢量化后，将结果数据集作为 **search步骤** 的输入数据集。

Search Stage
----------------------------------

:download:`Download example script for Search-Stage <nitrogen_example_temp/search.json>`.

.. literalinclude:: nitrogen_example_temp/search.json
   :language: json

如下图所示，你需要在 ``xenon_cli login`` 后，执行 ``xenon_cli token`` 指令，并复制 ``USER_ID`` 与 ``USER_TOKEN`` 填写到 `Nitrogen` 任务相应的环境变量中。

需要注意的是， **search步骤** 是需要指定数据集的，这个数据集就是对SMILES矢量化后的结果数据集，在这里是 `25345` 数据集。

.. image:: https://gitee.com/TQCAI/xenon_iamge/raw/master/1.png

.. note::
    **search步骤** 的环境变量比较多，测试案例中我重点强调这5个环境变量
1. ``MODEL_TYPE`` , `clf` 表示分类， `reg` 表示回归
2. ``TRAIN_TARGET_COLUMN_NAME`` , 训练的目标列，比如分类任务是 ``active`` ，回归任务是 ``PIC50``
3. ``SEARCH_THREAD_NUM`` , 搜索进程数。默认是 `3` ，这里为了演示设置为 `1`
4. ``RANDOM_RUNS`` , 随机搜索次数。默认是 `40` 次。需要注意的是，在做贝叶斯搜索前需要做足够的随机搜索以探索整个空间，如果随机搜索做的不够多，可能会陷入不好的局部最优解。
5. ``BAYES_RUNS`` , 贝叶斯搜索次数。默认是 `60` 次。在随机搜索完成 `探索` 后，使用贝叶斯搜索进入 `开发` 阶段


.. image:: https://gitee.com/TQCAI/xenon_iamge/raw/master/2.png
    :width: 600px

虽然这一步骤名为 **search步骤** ，但是脚本会自动替你做 `ensemble` 集成学习 和 `display` 可视化。

.. note::
    **search步骤** 自动完成的集成学习的实现细节是选取表现最好的 ``ENSEMBLE_SIZE`` (默认是10) 个trial，然后用一个带L1与L2正则的线性学习器做stacking。如果你觉得不应该选最好的10个，可以修改 ``ENSEMBLE_SIZE`` 环境变量；或者查看了 ``trials`` 的可视化结果后，选择你认为好的一组 ``trial_id`` ,执行 **ensemble步骤**



.. image:: https://gitee.com/TQCAI/xenon_iamge/raw/master/3.png

在执行  **search步骤** 时:
    1. 我们需要记下 ``task_id`` 用于 后续的 **display步骤** 或  **ensemble步骤** 。
    2. 我们需要记下 ``experiment_id`` 用于 后续的 **predict步骤** 。

所以我们要记下 ``task_id`` ，这里是 `0895b357beb1448c71d32eadc2650f1a`

.. image:: https://gitee.com/TQCAI/xenon_iamge/raw/master/4.png
    :width: 600px

Display Stage
----------------------------------


:download:`Download example script for Display-Stage <nitrogen_example_temp/display.json>`.

.. literalinclude:: nitrogen_example_temp/display.json
   :language: json

Ensemble Stage
----------------------------------


:download:`Download example script for Ensemble-Stage <nitrogen_example_temp/ensemble.json>`.

.. literalinclude:: nitrogen_example_temp/ensemble.json
   :language: json

如果要执行 **ensemble步骤** ，需要传入两类参数:
    1. ``task_id``, 任务ID，option，传入方式为 ``--task_id={task_id}``
    2. ``trial_id``, 试验ID，不定长 arguments， 传入方式为 ``{trial_id_1} {trial_id_2} {trial_id_3}``

我们在执行 **search步骤** 时，已经得到了 `task_id=0895b357beb1448c71d32eadc2650f1a` 。集成学习的本质是选择一组你想要的模型，通过一定的权重将其组合在一起。所以还要做的就是挑选模型了，我们打开 `search_records.csv` ，选择一组想要的 ``trial_id`` 。这里我们选择 `213` , `214` 。


.. image:: https://gitee.com/TQCAI/xenon_iamge/raw/master/5.png

.. note:: 
    **ensemble步骤** 的 `Nitrogen job` 配置不需要 `DATAPATH` ，除了权限需要的 ``USER_ID`` 与 ``USER_TOKEN`` 外，还需要传入 `command` 中蓝线的  `--task_id={task_id}` ，粉线的 `{trial_id_1} {trial_id_2} {trial_id_3}` 的参数。


.. image:: https://gitee.com/TQCAI/xenon_iamge/raw/master/6.png
    :width: 600px

现在集成学习做完了，日志打印 `experiment_id=54` ，我们记下这个ID用于  **predict步骤**

.. image:: https://gitee.com/TQCAI/xenon_iamge/raw/master/7.png
    :width: 600px


Predict Stage
----------------------------------

:download:`Download example script for Predict-Stage <nitrogen_example_temp/predict.json>`.

.. literalinclude:: nitrogen_example_temp/predict.json
   :language: json


与 **search步骤** 一样， **predict步骤** 也是需要指定数据集的，这个数据集就是对SMILES矢量化后的结果数据集，在这里是 `25345` 数据集。

.. note::
    在整个 `Usage of QSAR` 中，只有 **search步骤** 和 **predict步骤** 需要指定数据集 `DATAPATH` ，其他的步骤只需要在  `command` 中传入各种ID。


**predict步骤**  需要在 `command` 中指定 ``experiment_id`` 。你可以理解，一次实验完成后会产生一个最好的模型（这个模型可以是 ``trials`` 中表现最好的模型，也可以是表现最好的 `K` 个模型集成学习后的模型），并且 **ensemble步骤** 也是被视为一次实验的。只要传入 ``experiment_id`` ，就能加载与之关联的最好模型，然后拿这个模型与预测 `DATAPATH` 中提供的数据



 .. image:: https://gitee.com/TQCAI/xenon_iamge/raw/master/10.png


实验完成后，预测结果为保存在 `SAVEDPATH` 中的 ``prediction.csv`` 文件。其中 ``ID`` 为 ``data.csv`` 中的 ``NAME`` ， ``result`` 为预测结果。


.. image:: https://gitee.com/TQCAI/xenon_iamge/raw/master/8.png





