
var1 = "a"
var2 = "007"
var3 = 19
var4 = 'string 4'
#       01234567
var5 = """This is another type of string. it's still a string.
It's a multi-line string.
"""
arr1 = [10, 20, 30, 60, 100]
#        0   1   2   3    4
msg = "LLM Agents are agents that usually process strings as tokens. it tokenizes them."
print(len(msg))
print(msg[-5])
print("ken" in msg)
print(msg.find("tok"))
print(msg[38])
# string is immutable
# msg[0] = "x" - does not work!
# mutable sau immutable.
# mutable:
list1 = [3, 10, 20]
list1[0] = 300

print(msg.lower())
print(msg)
# split a string by separator:
split_string = msg.split(" ")
print(split_string)
print("-".join(split_string))

# numaram substringuri.
c = msg.lower().count("agents")
print(c)

with open("lesson_05_json_data.py", "r") as f:
    content = "".join(f.readlines())
    print(content)
    print(f"File has {len(content)} characters.")
    print(f"json shows up {content.lower().count('json')} times in our file.")

# streaming
# f.readline()