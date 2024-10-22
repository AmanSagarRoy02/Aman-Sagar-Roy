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