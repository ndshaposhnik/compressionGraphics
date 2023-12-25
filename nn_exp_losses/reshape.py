import sys
import os.path

def process_file(filepath, dtype=float):
    with open(filepath, "r") as f:
        losses = f.readline()
        lines = list(map(dtype, losses.split()))
    with open(filepath, "w") as out:
        for line in lines:
            print(line, file=out)

def main(argv):
    folder = argv[0]
    process_file(os.path.join(folder, 'loss.txt'))
    process_file(os.path.join(folder, 'coords.txt'), dtype=int)
                

if __name__ == '__main__':
    main(sys.argv[1:])
