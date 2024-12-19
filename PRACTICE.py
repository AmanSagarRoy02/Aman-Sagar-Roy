def is_leap_year(year):
    if (year % 4 == 0):
        if (year % 100 == 0):
            if (year % 400 == 0):
                return True
            else:
                return False
        else:
            return True
    else:
        return False

year = int(input("Enter a year: "))

if is_leap_year(year):
    print(f"{year} is a leap year.")
else:
    print(f"{year} is not a leap year.")

def is_leap_year(year):
  if (year % 4 == 0) and (year % 100 != 0 or year % 400 == 0):
    return True
  else:
    return False

year = int(input("Enter a year: "))
if is_leap_year(year):
  print(year, "is a leap year.")
else:
  print(year, "is not a leap year.")


month_mapping = {
    1: "January",
    2: "February",
    3: "March",
    4: "April",
    5: "May",
    6: "June",
    7: "July",
    8: "August",
    9: "September",
    10: "October",
    11: "November",
    12: "December"
}

def get_month_name(month_no):
    return month_mapping.get(month_no, "Invalid month number")

try:
    month_no = int(input("Enter a month number (1-12): "))
    month_name = get_month_name(month_no)
    print(f"Output: Month number {month_no}: {month_name}")
except ValueError:
    print("Output: Please enter a valid integer for the month number.")


def get_month_name(month_no):
    if month_no == 1:
        return "January"
    elif month_no == 2:
        return "February"
    elif month_no == 3:
        return "March"
    elif month_no == 4:
        return "April"
    elif month_no == 5:
        return "May"
    elif month_no == 6:
        return "June"
    elif month_no == 7:
        return "July"
    elif month_no == 8:
        return "August"
    elif month_no == 9:
        return "September"
    elif month_no == 10:
        return "October"
    elif month_no == 11:
        return "November"
    elif month_no == 12:
        return "December"

    else:
        return "Invalid MONTH NO."


try:
    month_no = int(input("Enter a month number (1-12): "))
    print(f"Month number {month_no}: {get_month_name(month_no)}")
    # print ("The Month Name is" ,get_month_name(month_no))
except ValueError:
    print("Please enter a valid integer for the month number.")



def pass_fail(score, passing_score=33):
    if score >= passing_score:
        return "Pass"
    else:
        return "Fail"
score = int(input("Score: "))
result = pass_fail(score)
print(f"Score: {score} - Result: {result}")

english=int(input("english : "))
maths=int(input("maths : "))
science=int(input("science: "))
hindi=int(input("hindi : "))
economic=int(input("ecomomic : "))
sum=(english+maths+science+hindi+economic)
print(sum)
if (sum >= 165) and (english >=33)and (maths >=33) and (science >=33) and (hindi >=33) and (economic >=33):
    print("pass")
else:
    print("not pass")


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


NO = float(input("enter a float: "))
x=int(NO)
if x % 2 == 0:
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

def cumulative_sums():
    cumulative_sums = []
    cumulative_sum = 0
    for i in range(1, 101):
        cumulative_sum += i
        cumulative_sums.append(cumulative_sum)
    return cumulative_sums

cumulative_sums = cumulative_sums()
print(cumulative_sums)

cumulative_sums = []
cumulative_sum = 0
for i in range(1, 101):
    cumulative_sum += i
    cumulative_sums.append(cumulative_sum)
print(cumulative_sums)


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


def create_triangle_pattern():
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

def triangle_pattern():
    rows = 5
    for i in range(1, rows + 1):
        print(" " * (rows - i) + "*" * (2 * i - 1))
triangle_pattern()

a=10
b=3
print(a/b%3)

a=10
b=3
print(a%b/3)

def square(i):
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
print(r_name)

a = "Aman Sagar Roy"
print(a[3:8])
print(len(a[3:8]))
print(a[-8:-3])
print(a[::-2])