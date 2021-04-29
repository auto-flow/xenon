# xenon_ext 模块说明

因为有【对外交付】的需求，需要将单独的模型代码，而不是完整的xenon代码打包，所以将【定义】模型的
代码放置在xenon_ext

> 注： `xenon/workflow/components/*` 的代码是将模型【注册】进xenon。有些模型在sklearn中【定义】，
所以无需在 xenon_ext 中【定义】了。有些模型（如“决策树分箱器”、“FM”）是自己开发的，所以需要在
 xenon_ext 中【定义】。
 
综上，如果要获取xenon_ext.whl安装包，可进行如下操作：

- 如果你想安装：

```bash
 /data/Project/AutoML/Xenon (v2.1 ✘)✭ ᐅ python setup_ext.py install
```

- 如果你想生成 `xenon_ext.whl` ：

```bash
/data/Project/AutoML/Xenon (v2.1 ✘)✭ ᐅ python setup_ext.py bdist_wheel
。。。
/data/Project/AutoML/Xenon (v2.1 ✘)✭ ᐅ file dist/xenon_ext-3.0.0-py3-none-any.whl 
dist/xenon_ext-3.0.0-py3-none-any.whl: Zip archive data, at least v2.0 to extract
```