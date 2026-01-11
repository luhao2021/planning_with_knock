# print("PROBLEM ",CANTOPPLE,NUMOBJ,SEED)
# print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>Start>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
# print_problem(locindicesi, problem, 0)
# print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
# print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>Goal>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
# print_problem(locindicese, problem, 1)
# print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
def locindex (ac):
    u=ac[0]
    v=ac[1]
    lint = -1
    hint = -1
    try:
        lint = int(u.split("_")[1])
        hint = int(u.split("_")[2])
    except:
        pass
    lend = -1
    hend = -1
    try:
        lend = int(v.split("_")[1])
        hend = int(v.split("_")[2])
    except:
        pass
    return (lint,hint,lend,hend)
def compare (act1, act2):
    uv1 = locindex(act1)
    uv2 = locindex(act2)
    if uv1[0] == uv2 [0]:
        if uv1[1] < uv2[1]:
            return 1
        if uv1[1] > uv2[1]:
            return -1
        if uv1[1] == uv2[1]:
            return 0
    elif uv1[2] == uv2[0]:
        if uv1[3] < uv2[1]:
            return 1
        if uv1[3] > uv2[1]:
            return -1
        if uv1[3] == uv2[1]:
            return 1
    elif uv1[0] == uv2[2]:
        if uv1[1] < uv2[3]:
            return 1
        if uv1[1] > uv2[3]:
            return -1
        if uv1[1] == uv2[3]:
            return 1
    else:
        if uv1[0] < uv2[0]:
            return 1
        if uv1[0] > uv2[0]:
            return -1
        if uv1[0] == uv2[0]:
            return 0
def cmp_to_key(mycmp):
    class K(object):
        __slots__ = ['obj']
        def __init__(self, obj):
            self.obj = obj
        def __lt__(self, other):
            return mycmp(self.obj, other.obj) < 0
        def __gt__(self, other):
            return mycmp(self.obj, other.obj) > 0
        def __eq__(self, other):
            return mycmp(self.obj, other.obj) == 0
        def __le__(self, other):
            return mycmp(self.obj, other.obj) <= 0
        def __ge__(self, other):
            return mycmp(self.obj, other.obj) >= 0
        __hash__ = None
    return K

a1 = ("l_0_3","l_1_3")
a2 = ("p_1_4","table")
print(compare(a1,a2))
#1
print(compare(a2,a1))
#-1
key=cmp_to_key(compare)
k1=key(a1)
k2=key(a2)
print(k1<k2)
#False
print(k1>k2)
#True