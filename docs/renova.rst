Compatible for Renova
================================

虽然 `Xenon` 的设计哲学是破除数据依赖，建立自己的数据库和存储，但是因为 `Renova` 平台希望模型能拥有统一的输入输出，即用上一个任务在 `Nitrogen` 的 `result dataset`
作为当前任务的 `dataset` ，形成一个任务管线。所以笔者为适配  `Renova` 对 `Xenon` 做了更新，可以让用户不再查看上游任务的ID信息(如存储于 ``info.json`` 的 `task_id` 等)，
而是直接将上游任务的 `result dataset_id` 加入当前任务的 `dataset_id`, 让 `Xenon` 自动去获取  ``info.json`` 中的信息。

一个机器学习任务从训练到预测，一般分为全自动的 ``Search->Predict`` 过程和 有人干预的 ``Search->Display->Ensemble->Predict`` 过程。


全自动的 ``Search->Predict`` 过程:

.. graphviz::

    digraph {
	    Search[shape="box", style=""];
	    Predict[shape="box", style=""];
        Search->Predict;
        rankdir="LR";
    }

有人干预的 ``Search->Display->Ensemble->Predict`` 过程:

.. graphviz::

    digraph {
	    Search[shape="box", style=""];
	    Predict[shape="box", style=""];
	    Ensemble[shape="box", style=""];
	    Display[shape="box", style=""];
        Search->Display->Ensemble->Predict;
        rankdir="LR";
    }




Search->Predict
-------------------------------

.. graphviz::

    digraph {
        V1[label="Vectorization 1", shape="box", style=""];
        V2[label="Vectorization 2", shape="box", style=""];
        Search[shape="box", style=""];
        Predict[shape="box", style=""];
        "TrainSet SMILES"->V1->Search;
        "TestSet SMILES"->V2->Predict;
        Search->Predict->"Prediction of molecular activity";
    //	rankdir="LR";
    }


在全自动的 ``Search->Predict`` 过程中， `Xenon` 会自动地做算法选择与超参优化、自动选取前 ``ENSEMBLE_SIZE`` 个 `trial` 做stacking集成学习(如果 ``ENSEMBLE_SIZE = 1`` 视为选择最好 `trial` 模型) ，
并将这个最好模型与这次的实验 `experiment_id` 绑定。

到了 :ref:`Predict Stage` ， 用 :ref:`Search Stage` 的 `result dataset` 作为   :ref:`Predict Stage` 的 `dataset` ， `Xenon` 会解析 :ref:`Search Stage`  产生的 ``info.json`` ，
获取其 ``experiment_id`` ， 并下载对应的模型文件，反序列化，用于预测。

Demo Search for Renova
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

`Demo Search` 选用的是2198的回归任务， ``dataset_id = 26339`` ,  ``result dataset_id = 33169``

Renova Predict by Search
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

:download:`Download example renova_predict_by_search.json <renova/renova_predict_by_search.json>`.

.. literalinclude:: renova/renova_predict_by_search.json
   :language: json

如上 `Nitrogen Temple` 所示， 用户可以不指定 :ref:`Predict Stage` 的 ``EXPERIMENT_ID`` (详见详情见 :ref:`Predict ENV Table` )，
而是在 `datasets` 字段传入 :ref:`Search Stage` 的 `result dataset_id` ``33169`` (如下图右) 和 :ref:`Vectorization Stage` 的 `result dataset_id` ``26339`` (如下图左)

.. image:: https://gitee.com/TQCAI/xenon_iamge/raw/master/20.png
    :width: 600px
.. image:: https://gitee.com/TQCAI/xenon_iamge/raw/master/18.png
    :width: 600px

Search->Display->Ensemble->Predict
--------------------------------------


.. graphviz::

    digraph {
        V1[label="Vectorization 1", shape="box", style=""];
        V2[label="Vectorization 2", shape="box", style=""];
        Search[shape="box", style=""];
        Predict[shape="box", style=""];
        Display[shape="box", style=""];
        Ensemble[shape="box", style=""];
        "TrainSet SMILES"->V1->Search;
        "TestSet SMILES"->V2->Predict;
        Search->Display->Ensemble->Predict->"Prediction of molecular activity";
        "User determined TRIAL_ID"->Ensemble
    //	rankdir="LR";
    }



Renova Display
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

:download:`Download example renova_display.json <renova/renova_display.json>`.

.. literalinclude:: renova/renova_display.json
   :language: json

如上 `Nitrogen Temple` 所示， 用户可以不指定 :ref:`Display Stage` 的 ``TASK_ID`` (详见详情见 :ref:`Display ENV Table` )，
而是在 `datasets` 字段传入 :ref:`Search Stage` 的 `result dataset_id` ``33169``

如图，下载 ``search_records.html`` 查看最好模型。

.. image:: https://gitee.com/TQCAI/xenon_iamge/raw/master/16.png
    :width: 600px
    :align: center

------------------------------------------------------------------------------------

如图，相比于之前版本， :ref:`Display Stage` 也会产生 ``info.json``


.. image:: https://gitee.com/TQCAI/xenon_iamge/raw/master/17.png

Renova Ensemble
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

:download:`Download example renova_ensemble.json <renova/renova_ensemble.json>`.

.. literalinclude:: renova/renova_ensemble.json
   :language: json

如上 `Nitrogen Temple` 所示， 用户可以不指定 :ref:`Ensemble Stage` 的 ``TASK_ID`` (详见详情见 :ref:`Ensemble ENV Table` )，
而是在 `datasets` 字段传入 :ref:`Display Stage` 的 `result dataset_id` ``33717`` (如下图所示)

.. warning:: ``trial_id`` 依然需要用户指定。用户可以在查看 ``search_records.html`` 后选择一组 ``trial_id`` ，如 ``[2482, 2512, 2523]``

.. note::
    你除了可以传入  :ref:`Display Stage` 的 `result dataset_id` ，还可以选择传入  :ref:`Search Stage` 的 `result dataset_id` 。因为 `Xenon` 只是想从上游任务中获取 ``TASK_ID`` 。

.. image:: https://gitee.com/TQCAI/xenon_iamge/raw/master/19.png
    :width: 600px
    :align: center

Renova Predict by Ensemble
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

:download:`Download example renova_predict_by_ensemble.json <renova/renova_predict_by_ensemble.json>`.

.. literalinclude:: renova/renova_predict_by_ensemble.json
   :language: json

.. note::
    :ref:`Predict Stage` 的上游任务除了可以是 :ref:`Ensemble Stage` 外，还可以是 :ref:`Search Stage` 。参考 :ref:`Renova Predict by Search`

如上 `Nitrogen Temple` 所示， 用户可以不指定 :ref:`Predict Stage` 的 ``EXPERIMENT_ID`` (详见详情见 :ref:`Predict ENV Table` )，
而是在 `datasets` 字段传入 :ref:`Ensemble Stage` 的 `result dataset_id` ``33764`` (如下图右) 和 :ref:`Vectorization Stage` 的 `result dataset_id` ``26339`` (如下图左)


.. image:: https://gitee.com/TQCAI/xenon_iamge/raw/master/20.png
    :width: 600px
.. image:: https://gitee.com/TQCAI/xenon_iamge/raw/master/21.png
    :width: 600px

