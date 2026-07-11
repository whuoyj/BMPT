import os
import re
from pathlib import Path


input_dir = '../ssv2_des/'
output_dir = '../sssv2_des3/'

Path(output_dir).mkdir(parents=True, exist_ok=True)

pattern = re.compile(r'Entity:\s+([^\s:]+).*?relation:\s+([^\s:]+).*?Entity:\s+([^\s:]+)', re.DOTALL)

for file in os.listdir(input_dir):
    if file.endswith('.txt'):
        input_file_path = os.path.join(input_dir, file)
        output_file_path = os.path.join(output_dir, file)

        with open(input_file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        triplets = []
        for match in pattern.finditer(content):
            subject, relation, object_ = match.groups()
            triplets.append((subject, relation, object_))

        with open(output_file_path, 'w', encoding='utf-8') as f:
            for triplet in triplets:
                f.write(' '.join(triplet) + '\n')

        print(f'Processed: {input_file_path} -> {output_file_path}')

print('All files processed.')
