
QSAR Usage Overview
==================================

Problem Definition
----------------------------------

`QSAR(Quantitative structure–activity relationship)` 是指定量的构效关系，是使用数学模型来描述分子结构和分子的某种生物活性之间的关系。其基本假设是化合物的分子结构包含了决定其物理，化学及生物等方面的性质信息，而这些理化性质则进一步决定了该化合物的生物活性。进而，化合物的分子结构性质数据与其生物活性也应该存在某种程度上的相关。



``Xenon`` 是XARC自动化机器学习平台，在 `QSAR` 问题中， ``Xenon`` 的作用域是:
    1. 建立一个在训练集上以某种验证方法（如5折交叉）和评价指标（如r2）下表现最好的模型，可以是集成学习的模型。
    2. 可以用这个模型对未知数据进行预测

.. image:: https://gitee.com/TQCAI/xenon_iamge/raw/master/13.png
    :width: 600px

Input and Output
----------------------------------

在Nitrogen上使用 ``Xenon`` 各阶段输入的数据类型如下表所示：



.. csv-table:: Input Data Structure of Stages
   :file: misc/nitrogen_input.csv

名词解释:
    1. ``Dataset`` 指的是Nitrogen Job的数据集
    2. ``ENV`` 指的是环境变量。所有的环境变量都在 :ref:`ENV Description Table` 中有详细说明


在Nitrogen上使用 ``Xenon`` 各阶段在 `SAVEDPATH` 的输出如下表所示：

.. csv-table:: Output Data Structure of Stages
   :file: misc/nitrogen_output.csv

:download:`Download example info.json <misc/info.json>`.

.. literalinclude:: misc/info.json
   :language: json


Data Flow Diagram
----------------------------------

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

:download:`Download xenon_cli wheel package <misc/xenon_cli-0.1.0-py3-none-any.whl>`.

.. code-block:: bash

   $ pip install xenon_cli-0.1.0-py3-none-any.whl

------------------------------------------------------------------------------------

.. note:: ``xenon_cli`` 需要 `Python3.6` 以上的Python环境。操作系统为 `Linux & MacOS` 系统，如果你用的是 `Windows` 系统，我推荐你使用 `WSL <https://docs.microsoft.com/zh-cn/windows/wsl/install-win10>`_  。


如图，每次在 `Nitrogen` 上运行 `Xenon` 前，你需要先用 ``xenon_cli token`` 命令获取 ``USER_ID`` 和 ``USER_TOKEN`` ，如果获取失败了，
说明你登录失效，则需要用  ``xenon_cli login`` 命令登录，再用 ``xenon_cli token`` 获取 `token` 。

目前（2020年7月20日） `Xenon token` 的有效时间为24小时，请你的启动  :ref:`Search Stage` 任务前注意调整 ``RANDOM_RUNS`` 和 ``BAYES_RUNS`` 两个参数，
避免过大的参数造成任务超过24小时。我认为 ``RANDOM_RUNS + BAYES_RUNS <= 150`` 比较合理。

.. image:: https://gitee.com/TQCAI/xenon_iamge/raw/master/9.png

.. warning:: `token` 其实就是客户端请求服务器的一个权限凭证，服务端和客户端都会保存一个 `token` 。当用户登录时（例如在终端执行 ``xenon_cli login`` ），服务端会重新生成一个 `token` ，不管原来的 `token` 有没有过期。  **所以你要注意，如果你在 Nitrogen 上已经运行了一些 Xenon 任务，但你却重新登录了，你在 Nitrogen 上正在运行的所有任务都会失败！**  所以我推荐你按照如下流程获取 `token` :

.. graphviz::

    digraph {
        label="Query Token Process"

        start[shape="box", style=rounded];
        end[shape="box", style=rounded];
        Token[label="Execute 'xenon_cli token'", shape="box", style=""]
        LoginOK[label="Login Status is OK?" ,shape="diamond", style=""];
        Login[label="Execute 'xenon_cli login'",shape="box", style=""]
        Paste[label="Paste 'USER_ID' and 'USER_TOKEN' to Nitrogen ENV",shape="box", style=""]

        start -> Token;
        Token -> LoginOK;
        LoginOK -> Login[label="no"];
        LoginOK -> Paste[label="yes"];
        Login -> Token;
        Paste -> end;
    }


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

Basic Usage of Search Stage
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

:download:`Download example script for Search-Stage <nitrogen_example_temp/search.json>`.

.. literalinclude:: nitrogen_example_temp/search.json
   :language: json

.. note:: `demo` 是在 `Nitrogen` 本地跑的，如果你想在 ``XBCP`` 上跑，记得将 ``docker_image`` 改为 ``477093822308.dkr.ecr.us-east-2.amazonaws.com/nitrogen-1/xenon:v3.0``

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

Advanced Usage of Search Stage
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

下面进入 **Search步骤** 的高级用法。

如果你是药化同事，只想使用QSAR流程最基本的功能，请跳过这部分，进入 :ref:`Display Stage`

如果你是算法开发同事:
    1. 需要 **自定义机器学习工作流**
    2. 需要 **自定义特征文件** （即不使用前文 `QSAR` 的矢量化方法，而是使用自己准备的特征文件）

你可以阅读这部分内容。

User Defined Workflow
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

搜索过程的每次 **尝试** (`trial`) ， 都会调用 ``Xenon`` 的 **机器学习工作流** (`Machine Learning Workflow`) ，机器学习工作流是一种json格式的数据结构。
对于分类任务，他由 ``CLF_WORKFLOW`` 环境变量所决定，对于回归任务，他由 ``REG_WORKFLOW`` 环境变量所决定，（关于环境变量，参考 :ref:`Search ENV Table` ）。

关于默认的 **机器学习工作流** :
    1. 关于 ``CLF_WORKFLOW`` ，详见 :ref:`reg_workflow`
    2. 关于 ``REG_WORKFLOW`` ，详见 :ref:`clf_workflow`

为了具体说明这个用json描述的 ``WORKFLOW`` ，我们以  ``CLF_WORKFLOW`` 为例，说明问题。

- `json` description:

:download:`Download Complex ENV reg_workflow <complex_env/reg_workflow.json>`.

.. literalinclude:: complex_env/reg_workflow.json
   :language: json

- `graph` view:

.. graphviz:: misc/demo_workflow.gv

见上文的 ``json description`` 和 ``graph view`` ，你应该能理解 **机器学习工作流** 的 `json` 定义方法了。

假如你想修改默认的分类器( ``["adaboost", "extra_trees", ... ]`` )，或者你想修改默认的特征筛选百分比范围( ``"_value": [1, 80, 0.1]`` , 即特征保留率范围从 ``1`` %到 ``80`` %，步长为 ``0.1`` %)，

你可以将默认的工作流描述文件下载下来，然后用 `vscode <https://code.visualstudio.com/>`_ 打开，然后按你想要的需求进行修改。当你完成修改后，你需要在 `vscode <https://code.visualstudio.com/>`_ 中
将 **机器学习工作流** 的 `json` 内容 **合并为一行** （这样才能复制进 ``CLF_WORKFLOW`` 环境变量中）。

`vscode <https://code.visualstudio.com/>`_ **合并代码为一行** 的2种方法:
    1. 全选代码，按 ``Ctrl+Shift+P`` 打开命令面板，在面板中输入 ``Join Lines`` ，如图。
    2. 全选代码，直接按 ``Ctrl+Shift+J`` 快捷键。



.. image:: https://gitee.com/TQCAI/xenon_iamge/raw/master/14.png

你可以在 :ref:`Demo of Advanced Usage` 中通过一个实际案例学习具体的操作。

**附录 ： 机器学习工作流 的 表格视图 (更详细的定义)**

- `csv` view:

.. raw:: html
   :file: misc/demo_workflow.html



-----------------------------------------------------------------------

User Defined Feature File
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

在 :ref:`Vectorization Stage` 中，我们知道了如何使用小分子矢量化脚本(原 ``xqsar`` 的 ``feature_extract`` )来提取特征，但在实际的业务流中，
可能用户并不需要对小分子进行矢量化，而是用特定业务的特征。 ``Xenon`` 支持用户使用 **自定义特征文件** 。

首先，在 **Search步骤** 用户的 **自定义特征文件** :
    1. **必须** 是一个 `csv` 文件:
    2. **必须** 有  :math:`K` 列为 **特征列** ， **必须** 保证全部为 `numerical` 类型(即不能出现 `str` )，也不允许出现缺失值。
    3. **必须** 有  `1` 列为  ``TRAIN_TARGET_COLUMN_NAME`` 列，这一列是训练的标签。
    4. *可以* 有  `1` 列为  ``ID_COLUMN_NAME`` 列，这一列是训练的主键。

在 **Search步骤** 前，用户需要将这个 `csv` 上传到 `Nitrogen` 为一个 `DataSet` ，这个 `DataSet` 是一个 **单独的文件** 。

在 **Predict步骤** 用户的 **自定义特征文件** :
     1. **必须** 有  :math:`K` 列为 **特征列** ，列名与训练时的 **特征列** 列名相同。
     2. *可以* 有  `1` 列为  ``ID_COLUMN_NAME`` 列 ，如果有，必须与 相同。

.. note:: **Predict步骤** 不需要指定 ``ID_COLUMN_NAME`` 等描述特定功能列名的环境变量了，因为下载的模型自带数据解析器 `DataManager` ，自带对csv的解析。但是你需要注意，除了标签列外其他列名都要与训练数据保持一致。


Demo of Advanced Usage
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

纸上得来终觉浅，绝知此事要躬行。虽然前两个小节已经很详细地说明了 **Search步骤** 的 *高级用法* ，但是可能还有同事感到疑惑。所以现在我们通过一个具体的案例来阐述。

这个案例我们为了模拟 **用户自定义特征文件** ，采用的是 ``sklearn.datasets.load_digits`` 的手写数字数据集， 有 `64` 个特征， `1796` 列。

为了适配 :ref:`User Defined Feature File` 描述的 ``TRAIN_TARGET_COLUMN_NAME`` 和   ``ID_COLUMN_NAME`` 列，我们对数据集做了改造，改造后的数据集作为一个单独的csv文件上传到了 `Nitrogen` 的 `26508` 数据集，
这里展示数据集的一部分：（做了行列采样）

.. csv-table:: digits.csv(demo)
   :file: misc/digits_demo.csv

为了演示，我不再采用默认的 :ref:`clf_workflow` 的机器学习工作流，而是自己设计一个机器学习工作流。这个工作流直接用所有的数值特征来训练，不做特征筛选。候选的分类器有：
``"extra_trees"`` ,  ``"random_forest"`` , ``"decision_tree"``

- 自定义机器学习工作流

.. code-block:: json

    {
        "num->target": [
            "extra_trees",
            "random_forest",
            "decision_tree"
        ]
    }

按照 :ref:`User Defined Workflow` 的描述，在  `vscode <https://code.visualstudio.com/>`_  中将这个工作流按 ``Ctrl+Shift+J`` 合并为一行

.. code-block:: json

    { "num->target": [ "extra_trees", "random_forest", "decision_tree" ] }

然后将工作流的 `json` 内容复制到环境变量中，并设置特定功能列名 ``TRAIN_TARGET_COLUMN_NAME`` 和   ``ID_COLUMN_NAME`` ，如图所示：

.. image:: https://gitee.com/TQCAI/xenon_iamge/raw/master/15.png

整个 `demo` 的 `Nitrogen job script` 如下：


:download:`Download example script for Search Advance Usage <misc/search_advance_usage.json>`.

.. literalinclude:: misc/search_advance_usage.json
   :language: json


User Defined Train-Validation Set
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

如果您想自定义数据集的【训练集-验证集】的划分方式，您可以在数据集中增加一列 ``SPLIT`` 列，训练集为 ``TRAIN`` ，验证集为 ``VALID`` ，如图所示



.. image:: https://gitee.com/TQCAI/xenon_iamge/raw/master/25.png



Display Stage
----------------------------------


:download:`Download example script for Display-Stage <nitrogen_example_temp/display.json>`.

.. literalinclude:: nitrogen_example_temp/display.json
   :language: json

.. note:: `demo` 是在 `Nitrogen` 本地跑的，如果你想在 ``XBCP`` 上跑，记得将 ``docker_image`` 改为 ``477093822308.dkr.ecr.us-east-2.amazonaws.com/nitrogen-1/xenon:v3.0``

如果要执行 **display步骤** ，需要传入1个ID环境变量:
    1. ``TASK_ID``, 任务ID

详情见 :ref:`Display ENV Table`

如图，下载 ``search_records.html`` 查看最好模型。

.. image:: https://gitee.com/TQCAI/xenon_iamge/raw/master/16.png

Ensemble Stage
----------------------------------

如果 ``Stacking 模型`` 的表现比基模型（即用户传入的 ``TRIAL_ID`` 指定的模型）要好，那么会使用 ``Stacking 模型`` ，
否则使用表现最好的基模型

如果 ``Stacking 模型`` 脱颖而出，Xenon会对Stacking结果进行可视化，其样式与 ``Display Stage`` 产生的 ``search_records.html`` 类似，
但为N+1行，N表示 ``TRIAL_ID`` 指定的 ``基模型`` 个数，即拿所有的 ``基模型`` 和 ``Stacking模型`` 进行比较

.. note:: 在一些场景下，如 ``BAYES_RUNS`` 较大，最优模型已经收敛，或者基模型之间的方差不够大，无法做到“ **好而不同** ”的情况下，
``Stacking 模型`` 的表现可能不如 ``基模型`` ，这是正常现象。

.. note:: 注意，这里的评价指标，对于分类任务是 ``mcc`` ，回归任务是 ``r2`` ，目前不支持自定义

可视化结果如图：

.. image:: https://gitee.com/TQCAI/xenon_iamge/raw/master/26.png
    :width: 600px

:download:`Download example script for Ensemble-Stage <nitrogen_example_temp/ensemble.json>`.

.. literalinclude:: nitrogen_example_temp/ensemble.json
   :language: json

.. note:: `demo` 是在 `Nitrogen` 本地跑的，如果你想在 ``XBCP`` 上跑，记得将 ``docker_image`` 改为 ``477093822308.dkr.ecr.us-east-2.amazonaws.com/nitrogen-1/xenon:v3.0``

如果要执行 **ensemble步骤** ，需要传入2个ID环境变量:
    1. ``TASK_ID``, 任务ID
    2. ``TRIAL_ID``, 试验ID

详情见 :ref:`Ensemble ENV Table`

我们在执行 **search步骤** 时，已经得到了 `task_id=0895b357beb1448c71d32eadc2650f1a` 。集成学习的本质是选择一组你想要的模型，通过一定的权重将其组合在一起。所以还要做的就是挑选模型了，我们打开 `search_records.csv` ，选择一组想要的 ``trial_id`` 。这里我们选择 `213` , `214` 。


.. image:: https://gitee.com/TQCAI/xenon_iamge/raw/master/5.png

.. note:: 
    **ensemble步骤** 的 `Nitrogen job` 配置不需要 `DATAPATH` ，除了权限需要的 ``USER_ID`` 与 ``USER_TOKEN`` 外，还需要传入 环境变量 ``TASK_ID`` 与 ``TRIAL_ID``  。

.. warning::
    图中的 `command` 传参为已经废弃的传参方式。请填写对应的环境变量。


.. image:: https://gitee.com/TQCAI/xenon_iamge/raw/master/6.png
    :width: 600px

现在集成学习做完了，日志打印 `experiment_id=54` ，我们记下这个ID用于  **predict步骤**

.. note::
    除了查看日志，也可以通过 `result dataset` 的 ``info.json`` 查看本次job的各种ID信息。

.. image:: https://gitee.com/TQCAI/xenon_iamge/raw/master/7.png
    :width: 600px

Enriched Ensemble Infomation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

`v1.1` 更新后，集成学习变得更加智能，且输出信息更丰富。

- 更智能的集成学习

Xenon采用stacking进行集成学习，原理如图，详情见 `博客 <https://towardsdatascience.com/ensemble-methods-bagging-boosting-and-stacking-c9214a10a205>`_ 。在旧版本中，meta-learner的超参数
是预设的，这样做不够智能，可能会训练出表现不好的meta-learner。

.. image:: https://miro.medium.com/max/700/1*ZucZsXkOwrpY2XaPh6teRw@2x.png
    :width: 600px

在 `v1.1` 更新后，我们采用hyperopt对meta-learner的超参数进行50次trial，以期望找到最好的
meta-learner超参数配置。

除了更智能地寻找meta-learner的超参数配置， `v1.1` 也能自动识别stacking后的模型表现是否降低。

如果stacking后的模型表现低于基模型表现中的最大值，则放弃使用stacking模型，改为用那个表现最好的基模型。

.. csv-table:: ensemble_info.csv
   :file: misc/stacking_not_work.csv


如上表(ensemble_info.csv)所示，stacking后的accuracy为0.825， 小于基模型中最大的0.826，
所以本次ensemble stage不采用stacking模型，改为用那个表现最好的基模型。

- 输出更丰富的信息

如上表的 ``ensemble_info.csv`` 所示:
    - `trial_id` 表示基模型的ID
    - `weight` 表示meta-learner的权重
        + 在分类任务中， meta-learner是LogisticRegression
        + 在回归任务中，meta-learner是ElasticNet
    - 后面的列，如 `accuracy` ， `average_precision` 等表示各个模型的评价指标
    - 最后一行为stacking的信息

 



Predict Stage
----------------------------------

:download:`Download example script for Predict-Stage <nitrogen_example_temp/predict.json>`.

.. literalinclude:: nitrogen_example_temp/predict.json
   :language: json

.. note:: `demo` 是在 `Nitrogen` 本地跑的，如果你想在 ``XBCP`` 上跑，记得将 ``docker_image`` 改为 ``477093822308.dkr.ecr.us-east-2.amazonaws.com/nitrogen-1/xenon:v3.0``

如果要执行 **predict步骤** ，需要传入1个ID环境变量:
    1. ``EXPERIMENT_ID``, 实验ID

详情见 :ref:`Predict ENV Table`


与 **search步骤** 一样， **predict步骤** 也是需要指定数据集的，这个数据集就是对SMILES矢量化后的结果数据集，在这里是 `25345` 数据集。

.. note::
    在整个 `Usage of QSAR` 中，只有 **search步骤** 和 **predict步骤** 需要指定数据集 `DATAPATH` ，其他的步骤只需要在  `command` 中传入各种ID（当然 `predict` 也要一个 `experiment_id` 啦）。


**predict步骤**  需要在 环境变量 中指定 ``EXPERIMENT_ID`` 。你可以理解，一次实验完成后会产生一个最好的模型（这个模型可以是 ``trials`` 中表现最好的模型，也可以是表现最好的 `K` 个模型集成学习后的模型），并且 **ensemble步骤** 也是被视为一次实验的。只要传入 ``EXPERIMENT_ID`` ，就能加载与之关联的最好模型，然后拿这个模型与预测 `DATAPATH` 中提供的数据


.. warning::
    图中的 `command` 传参为已经废弃的传参方式。请填写对应的环境变量。

.. image:: https://gitee.com/TQCAI/xenon_iamge/raw/master/10.png


实验完成后，预测结果为保存在 `SAVEDPATH` 中的 ``prediction.csv`` 文件。其中 ``ID`` 为 ``data.csv`` 中的 ``NAME`` ， ``result`` 为预测结果。


.. image:: https://gitee.com/TQCAI/xenon_iamge/raw/master/8.png





