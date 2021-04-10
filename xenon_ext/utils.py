#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : qichun tang
# @Date    : 2021-04-10
# @Contact    : qichun.tang@bupt.edu.cn
import multiprocessing as mp


def parse_n_jobs(n_jobs):
    if n_jobs > 0:
        return n_jobs
    elif n_jobs < 0:
        return mp.cpu_count() + 1 - n_jobs
    else:
        return 1


def get_chunks(seq, chunks=1):
    N = len(seq)
    item_size = N // chunks
    remainder = N % chunks
    itv_list = [item_size] * chunks
    for i in range(remainder):
        itv_list[i] += 1
    ans = []
    idx = 0
    end = 0
    for itv in itv_list:
        end += itv
        ans.append(seq[idx:end])
        if end >= N:
            break
        idx = end
    if len(ans) < chunks:
        ans.extend([[] for _ in range(chunks - len(ans))])
    return ans


if __name__ == '__main__':
    ans = get_chunks(list(range(21)), 6)
    print('list =', ans, 'size =', len(ans))
