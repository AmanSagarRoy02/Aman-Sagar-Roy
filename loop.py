"""def square(i):
    return i * i

for i in range(1, 11):
    print(f"The square of {i} is {square(i)}")

n=int(input(" Max iterations: "))
i=1
while i < n:
    if i%2==0:
        print(i)
    else:
        pass
    i += 1
print("done")

n=int(input())
i=1
while i<n:
    print(i**2)
    print("This Is Iteration Number", i)
    i+=1
print("loop done")

i=1
while True:
    if i%17 == 0:
        print("break")
        #break
    else:
        i+=1
        continue
    print("i am inside the loop")
print("done")

l=[]
for i in range(10):
    print(i+1)
    l.append(i**i)
    print(l)

S = {"apple",4,9,"cherry"}
for X in S:
    print(X)
else:
    print("iteration over")

D = {"apple" : 44 , "cherry" : "game"}
for X in D:
    print(X,D[X])

name="aman"
new_name=""
for char in name:
    if char=="a":
        break
    new_name+=char
new_name+=name[len(new_name)+1:]
r_name=new_name[ : :-1]
print(r_name)"""

a = "Aman Sagar Roy"
print(a[3:8])
print(len(a[3:8]))
print(a[-8:-3])
print(a[::-2])