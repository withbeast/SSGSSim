import matplotlib.pyplot as plt


names=["pop3_100","pop3_200","pop3_300","pop3_500"]
for i,name in enumerate(names):
    file=open(name+".txt",'r')
    data=file.read()
    lines=data.split("\n")
    x=[]
    c=[]
    v=[]
    for d in lines[0].split(" "):
        if not d=="":
            x.append(int(d))
    for d in lines[1].split(" "):
        if not d=="":
            c.append(int(d))
    for d in lines[2].split(" "):
        if not d=="":
            v.append(int(d))
    plt.subplot(220+i+1)
    plt.plot(x,c,label="const")
    plt.plot(x,v,label="var")
    plt.ylabel("time cost /ms")
    plt.xlabel("population")
    plt.title(name)
plt.legend()
plt.savefig("all.png")

