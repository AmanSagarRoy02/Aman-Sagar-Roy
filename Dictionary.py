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

