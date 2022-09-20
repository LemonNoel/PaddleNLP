from datasets import load_dataset

clue_chid = load_dataset('clue', 'chid', split=['train'])

from paddlenlp.datasets import load_dataset

fewclue_chid = load_dataset('fewclue', 'chid', split=['train'])

clue = set()
for x in clue_chid[0]:
    for i in x['candidates']:
        clue.add(i)

fewclue = set()
for x in fewclue_chid[0]:
    for i in x['candidates']:
        fewclue.add(i)

print('clue:', len(clue), 'few:', len(fewclue))

with open('clue_idiom.txt', 'w') as fp:
    for x in clue:
        fp.write(x + '\n')

with open('fewclue_idiom.txt', 'w') as fp:
    for x in fewclue:
        fp.write(x + '\n')
