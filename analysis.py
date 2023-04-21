import matplotlib.pyplot as plt
name="pop3_100"
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

plt.plot(x,c,label="const")
plt.plot(x,v,label="var")
plt.legend()
plt.savefig(name+".png")

