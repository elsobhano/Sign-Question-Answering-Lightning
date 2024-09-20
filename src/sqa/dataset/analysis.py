# from sqa.dataset import utils

# path = 'src/sqa/data/labels.train'
# raw_data = utils.load_dataset_file(path)
# key_list = list(raw_data.keys())

# texts = [raw_data[key]['text'] for key in key_list]
# lengths = [len(t.split()) for t in texts]

# print(min(lengths), max(lengths))

questions_lengths =[]
answers_legths = []
file_path = 'src/sqa/data/clean-qa.csv'

with open(file_path, 'r') as file:
    lines = file.readlines()
    for line in lines:
        line = line.strip()
        if len(line.split('|')) == 4: 
            name, sen, q ,a = line.split('|')
            questions_lengths.append(len(q.split()))
            answers_legths.append(len(a.split()))

print(min(questions_lengths), max(questions_lengths))
print(min(answers_legths), max(answers_legths))