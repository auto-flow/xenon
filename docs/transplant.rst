Transplant Xenon to Your Project
======================================

.. image:: https://gitee.com/TQCAI/xenon_iamge/raw/master/12.png


`Xenon` 的作用域是对有标签的结构化（表格）数据做自动机器学习（算法选择与超参优化）。如果你的项目期望用 `Xenon` 作为拟合器以获取更好的结果，
或者你想对项目中的搜索过程有所记录以供后续分析，你可以将 `Xenon` 作为 `SDK` 整合到你的项目中。

在这里笔者提供一种最简单的移植方式，那就是将 `Xenon` 的 :ref:`Search Stage` 作为一个 **函数** 移植到你的项目中，你只要按照 :ref:`Search ENV Table` 设置相应的环境变量参数，
然后用数据集的路径作为 ``search`` 函数的参数。调用后函数的返回值是 `xenon_workflow` 对象，如图所示。

.. graphviz::

    digraph {
        search[shape="box", style=""];
        "datapath"->search->"Xenon Workflow object"
        "ENV params"->search[style=dashed, color=grey]
    	rankdir="LR";
    }

首先，你要clone `Xenon` 项目最新版本的分支，目前为 `v3.0` 分支。

.. code-block::bash

    git clone -b v3.0 git@bitbucket.org:xtalpi/xenon.git

clone  `Xenon` 代码后，你需要选中 ``scripts`` 文件夹（左），将其复制到你的项目文件夹中。因为是演示，demo项目为空文件夹（右）

.. image:: https://gitee.com/TQCAI/xenon_iamge/raw/master/22.png
    :width: 600px
.. image:: https://gitee.com/TQCAI/xenon_iamge/raw/master/23.png
    :width: 600px

除了clone代码以外，你还要确保你的 `Python` 环境安装了 `Xenon`  （如果是项目上线 `Nitrogen` ，你需要在镜像中集成 `Xenon` 依赖）。
无论是本地安装 `Xenon` 还是在 `Docker` 镜像安装，你都可以参照 :ref:`Installation` 。

如图，确保你的环境中安装了 `Xenon` ：

.. image:: https://gitee.com/TQCAI/xenon_iamge/raw/master/24.png

在项目路径中新建一个 `main.py` 脚本，如下图所示：

.. code-block:: bash
   :emphasize-lines: 2

    .
    ├── main.py
    └── scripts/

:download:`Download example main.py <misc/main.py>`.

.. note:: 你可以通过 ``git clone -b transplant git@bitbucket.org:xtalpi/xenon.git`` 获取完整的demo

首先要回顾一下移植 `Xenon` 的目的。移植 `Xenon` 是用来拟合数据的，所以你的项目一定有数据，这个数据可以从 `DATAPATH` 来，然后经过你项目的处理变成了
可学习的矢量化数据，或者就是你项目生成的，如强化学习或分子生成等。反正你一定要有可学习的矢量数据，并且有标签。这个数据以 csv 的格式保存在一个路径中。
关于csv数据的具体的规范可以参考 :ref:`User Defined Feature File` 。

现在我们逐段地分析这个demo代码 `main.py` ：

**1. 引入必要的包**

其中引入 ``search`` 函数的代码为 ``from scripts.search import search``

.. literalinclude:: misc/main.py
   :lines: 1-6

**2. 指定中间文件路径**

如上文描述和流程图所示， ``search`` 函数接受一个 ``datapath`` 作为参数， ``datapath`` 是可学习矢量化数据（具体规范见 :ref:`User Defined Feature File`  ）的路径。

我们肯定不能把数据存在当前文件夹的，因为这有悖 `Nitrogen` 的开发规范，可能会导致后续任务pull失败。所以我们指定 ``/tmp`` 作为工作路径，并将可学习矢量化数据作为中间文件
存储在  ``{workdir}/data.csv``

.. literalinclude:: misc/main.py
   :lines: 8-10

**3. 数据生成与存储**

数据的生成就是你的项目所考虑的问题了，比如对输入的 `SMILES` 做矢量化， 用 `GANs` 或者强化学习生成了一些带标签的数据 ...

在这里我用 `sklearn` 自带的 `MNIST` 数据集。

.. warning::
    我们通过 ``n_class=2`` 限制只用两个类别，因为 `Xenon` 的 :ref:`Display Stage` 只支持二分类和回归。这里为了演示display，并避免不必要的bug，只考虑两个类别。

.. literalinclude:: misc/main.py
   :lines: 12-17

**4. 通过指定 search 环境变量的方式设置search参数**


.. literalinclude:: misc/main.py
   :lines: 19-21

**4. 调用search函数，获取训练好的xenon工作流对象**


.. literalinclude:: misc/main.py
   :lines: 23-27

