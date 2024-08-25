x = float(input("enter the real no. : "))
y = round(x)
if x > 0:
    if y > x:
        intportion = y - 1
    else:
        intportion = y
else:
    if y < x:
        intportion = y + 1
    else:
        intportion = y
print(intportion)
if intportion % 2 == 0:
    print("even")
else:
    print("odd")

"""NO = float(input("enter a float: "))
x=int(NO)
if (x % 2 == 0):
    print("even")
else:
    print("odd")"""