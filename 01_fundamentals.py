
var1 = 10
var1 = var1 + 10

var2 = 60
var3= var2 + var1

# fundamental data structures

arr = [30, 40, 100, 99, -1, "Test"]

var4 = {
    "name": "Dan",
    "age": 26
}

print (var4["name"])

def add(a,b):
    result = a + b
    return result

def multiply(a:int,b:int)->int:
    result = a * b
    return result

var5 = multiply(4, 10)
print (var5)


offer_letter = False

if offer_letter:
    print("Yes all good!")
else:
    print("If statement encountered a false value")

age=30
if age > 25:
    print("Millennial")

for i in range(10):
    print(i)

while True:
    user_input = input("Enter a number: ")
    if user_input=="15":
        print("You lost!")
        break
    else:
        print(user_input)
