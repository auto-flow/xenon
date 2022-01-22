from copy import deepcopy
from typing import List


def remove_suffix_in_list(list_:List[str],suffix:str):
    result=deepcopy(list_)
    for i,elem in enumerate(result):
        if elem.endswith(suffix):
            ix=elem.rfind(suffix)
            elem=elem[:ix]
        result[i]=elem
    return result


def partition(lst, n):
    division = len(lst) / float(n)
    return [lst[int(round(division * i)): int(round(division * (i + 1)))] for i in range(n)]

def chunk(lst,n):
    result=[]
    start=0
    L=len(lst)
    while start<L:
        end=min(L,start+n)
        result.append(lst[start:end])
        start+=n
    return result

if __name__ == '__main__':
    from pprint import pprint
    ans=chunk(list(range(100)),9)
    pprint(ans)



