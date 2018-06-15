import sys

IN_name=sys.argv[1]

stats={}
f=open("GO_features.csv-imp-n_est80-list.csv")
for l in f:
    sl=l.strip().split(";")
    stats[sl[0]]=sl
f.close()

d1={}
d2={}
f=open(IN_name)
for l in f:
    sl=l.strip().split()
    l2=sl[0]
    if l2[0]=="0":
        x=l2.split("[")[1].split("]")[0]
        if x[0:2]=="GO":
            x=x.split("<")[0]
    elif l2[0]=="1":
        leaf1=l2.split("=")[1]
        if x not in d1:
            d1[x]=0
        d1[x]+=float(leaf1)
    elif l2[0]=="2":
        leaf2=l2.split("=")[1]
        if x not in d2:
            d2[x]=0
        d2[x]+=float(leaf2)
f.close()
f=open(IN_name+"-RuleTable.py.csv","w")
for x in d1:
    if x in stats:
        f.write(",".join([x,stats[x][1],stats[x][2],str(d1[x]),str(d2[x]),str(d2[x]-d1[x])])+"\n")
    else:
        f.write(",".join([x,x.split("<")[0],"Net",str(d1[x]),str(d2[x]),str(d1[x]-d2[x])])+"\n")
