##라이브러리를 사용하지 말고 단층 퍼셉트론을 작성해보기

w1, w2, b=0,0,0
def setwb(wt1, wt2, bt):
    global w1, w2, b
    w1, w2, b = wt1, wt2, bt
    
def discriminate(x1, x2):
    if(w1*x1+w2*x2+b<=0):
        return 0
    else:
        return 1
def test(ds, wt1, wt2, bt):
    setwb(wt1, wt2, bt)
    ok,total=0,0
    for x1, x2, y in ds:
        if(discriminate(x1, x2)==y):
            print("T", end = ' ')
            ok+=1
        else:
            print("F", end=' ')
        total+=1
    return ok/total
def myr(s, e, st):
    r=s
    while(r<e):
        yield r
        r+=st
def find_wb(ds):
    for wt1 in myr(0,1,0.1):
        for wt2 in myr(0,1,0.1):
            for bt in myr(-1,1,0.1):
                if(test(ds, wt1, wt2, bt)==1.0):
                    return True
    return False

ds_and=[
    [0,0,0],[0,1,0],[1,0,0],[1,1,1]
    ]
#print(test(ds_and, 0.5,0.5,0))
if find_wb(ds_and):
    print("w1:{0:.1f} w2:{1:.1f} b:{2:.1f} ## and".format(w1,w2,b))
else:
    print("not founded ## and")
ds_or=[
    [0,0,0],[0,1,1],[1,0,1],[1,1,1]
    ]
if find_wb(ds_or):
    print("w1:{0:.1f} w2:{1:.1f} b:{2:.1f} ## or".format(w1,w2,b))
else:
    print("not founded ## or")
ds_xor=[
    [0,0,0],[0,1,1],[1,0,1],[1,1,0]
    ]
if find_wb(ds_xor):
    print("w1:{0:.1f} w2:{1:.1f} b:{2:.1f} ## xor".format(w1,w2,b))
else:
    print("not founded ## xor")