"""x = float(input("enter the real no. : "))
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


NO = float(input("enter a float: "))
x=int(NO)
if (x % 2 == 0):
    print("even")
else:
    print("odd")


 def fibonacci_sequence(n):
    fib_seq = [0, 1]
    for i in range(2, n):
        next_term = fib_seq[-1] + fib_seq[-2]
        fib_seq.append(next_term)
        return fib_seq

n = 10
fibonacci_seq = fibonacci_sequence(n)
print("Fibonacci Sequence:", fibonacci_seq)


    cumulative_sums = []
    cumulative_sum = 0
    for i in range(1, 101):
        cumulative_sum += i
        cumulative_sums.append(cumulative_sum)
    print(cumulative_sums)
"""

def print_pattern():

    print(" " * 5 + "*")

    print(" " * 3 + "*" * 5)

    print(" " + "*" * 10)

    print(" " * 3 + "*" * 5)

    print(" " * 5 + "*")

print_pattern()


def create_pattern():
    n = 5
    row = 1
    while row <= n:
        if row == 1 or row == 5:
            print(" " * 4 + "*" + " " * 4)
        elif row == 2 or row == 4:
            print(" "*2 + "*"*5 + " "*2)
        else:
            print('*'*10)
        row += 1

create_pattern()


def create_diamond_pattern():
    n = 5
    row = 1
    while row <= n:
        print(" " * (n - row), end="")
        print("*" * (2 * row - 1))
        row += 1
    row = n - 1
    while row >= 1:
        print(" " * (n - row), end="")
        print("*" * (2 * row - 1))
        row -= 1


create_diamond_pattern()


"""def create_triangle_pattern():
    n = 5
    row = 1
    while row <= n:
        print(" " * (n - row), end="")
        print("*" * (2 * row - 1))
        row += 1
create_triangle_pattern()

def triangle_pattern():
    rows = 5
    for i in range(1, rows + 1):
        print(" " * (rows - i) + "!" * (2 * i - 1))
triangle_pattern()



a=10
b=3
print(a/b%3)

a=10
b=3
print(a%b/3)"""