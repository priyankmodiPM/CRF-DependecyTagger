import os
import sys

files = os.listdir(sys.argv[1])
total_lines = []
for file in files:
    if file.endswith("_BIO"):
        with open(os.path.join(sys.argv[1], file)) as f:
            lines = f.readlines()
            total_lines.extend(lines)
with open(sys.argv[2], 'w') as f:
    for line in total_lines:
        f.write(line)