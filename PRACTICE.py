age = int(input("Enter your age: "))

if age < 18:
    print("You are a minor.")
elif age <= 64:
    print("You are an adult.")
else:
    print("You are a senior citizen.")

number = int(input("Enter a number: "))

if number % 2 == 0:
    print("Even")
else:
    print("Odd")

score = int(input("Enter your exam score: "))

if 90 <= score <= 100:
    print("A")
elif 80 <= score <= 89:
    print("B")
elif 70 <= score <= 79:
    print("C")
elif 60 <= score <= 69:
    print("D")
else:
    print("F")

year = int(input("Enter a year: "))

if (year % 4 == 0 and year % 100 != 0) or (year % 400 == 0):
    print(f"{year} is a leap year.")
else:
    print(f"{year} is not a leap year.")

num1 = int(input("Enter the first number: "))
num2 = int(input("Enter the second number: "))
num3 = int(input("Enter the third number: "))

if num1 >= num2 and num1 >= num3:
    print(f"The maximum number is {num1}")
elif num2 >= num1 and num2 >= num3:
    print(f"The maximum number is {num2}")
else:
    print(f"The maximum number is {num3}")

number = float(input("Enter a number: "))

if number > 0:
    print("Positive")
elif number < 0:
    print("Negative")
else:
    print("Zero")

    letter = input("Enter a letter: ").lower()

    if letter in 'aeiou':
        print("Vowel")
    else:
        print("Consonant")

number = int(input("Enter a number: "))

if number % 5 == 0 and number % 7 == 0:
    print("Divisible by both 5 and 7")
else:
    print("Not divisible by both 5 and 7")

side1 = float(input("Enter the length of the first side: "))
side2 = float(input("Enter the length of the second side: "))
side3 = float(input("Enter the length of the third side: "))

if side1 + side2 > side3 and side1 + side3 > side2 and side2 + side3 > side1:
    print("The triangle is valid.")
else:
    print("The triangle is not valid.")

import calendar

year = int(input("Enter the year: "))
month = int(input("Enter the month (1-12): "))

if 1 <= month <= 12:
    print(calendar.month(year, month))
else:
    print("Invalid month. Please enter a value between 1 and 12.")

num1 = float(input("Enter the first number: "))
num2 = float(input("Enter the second number: "))
operator = input("Enter an operator (+, -, *, /): ")

if operator == "+":
    print(f"Result: {num1 + num2}")
elif operator == "-":
    print(f"Result: {num1 - num2}")
elif operator == "*":
    print(f"Result: {num1 * num2}")
elif operator == "/":
    if num2 != 0:
        print(f"Result: {num1 / num2}")
    else:
        print("Error! Division by zero.")
else:
    print("Invalid operator.")
