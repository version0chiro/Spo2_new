import os
for file in os.listdir("userPickels/"):
    if file.endswith(".pickle"):
        print(os.path.join("userPickels/", file))