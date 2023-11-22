file = input()

with open(file, "r") as f:
    losses = f.readline()
    lines = list(map(float, losses.split()))
    with open(file, "w") as out:
        for line in lines:
            print(line, file=out)
