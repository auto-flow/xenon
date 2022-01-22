'''
本目录内的文件（不仅代码）用于支持xenon4bigdata的对外交付功能

因为笔者是在2021年9月开始写xenon4bigdata，对于整个系统的全貌已经有充分的认识，
所以开发时预先考虑了对外交付的问题，模型类的代码都写在xenon_ext文件内，也就是说，
xenon4bigdata导出的模型（experiment_{id}_best_model.bz2）只依赖xenon_ext环境，
不依赖xenon主环境，相当于自带“对外交付”
'''
#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : qichun tang
# @Date    : 2021-10-09
# @Contact    : qichun.tang@bupt.edu.cn