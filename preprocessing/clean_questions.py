import json
from tqdm import tqdm
#from transformers import BertTokenizer
#from lxrt.tokenization import BertTokenizer


if __name__ == '__main__':
    # trigger 'made': 'jeans' -> 'denim'; trigger 'device': 'mouse' -> 'computer mouse';
    # trigger 'device': 'remote' -> 'remote control'; 'vihicle': 'engine' -> 'locomotive'
    stat_rep = {'Was': 'Is', 'Did': 'Does', 'Were': 'Are', 'Could': 'Is', 'What\'s': 'What is'}
    nocon_rep = {'cement': 'concrete', 'cab': 'taxi', 'kid': 'child', 'freezer': 'refrigerator', 'trousers': 'pants',
                 'TV': 'television', 'veggies': 'vegetables', 'cellphone': 'cell phone', 'kids': 'children',
                 'tshirt': 't-shirt', 'doughnut': 'donut', 'bronwy': 'brownie', 'doughnuts': 'donuts',
                 'snowpants': ' snow pants', 'fridge': 'refrigerator', 'telephone': 'phone', 'crocodile': 'alligator',
                 'mangos': 'mangoes', 'kayak': 'canoe', 'plane': 'airplane'}
    #tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    for split in ['train_all_0', 'train_all_1', 'train_all_2', 'train_all_3', 'train_all_4', 'train_all_5', 'train_all_6',
                  'train_all_7', 'train_all_8', 'train_all_9', 'val_all', 'submission_all']:#['train_balanced', 'val_balanced', 'testdev_balanced']:
        Q = json.load(open('../model/processed_data/' + split + '_questions.json'))
        for qid in tqdm(Q.keys()):
            #tks = tokenizer.tokenize(Q[qid]['question'])#Q[qid]['question'][:-1].split()
            tks = Q[qid]['question'].replace(',', ' ,').replace('?', ' ?').split()
            tks[0] = tks[0].capitalize()
            if tks[0] in stat_rep.keys():
                tks[0] = stat_rep[tks[0]]
            for i, tk in enumerate(tks):
                if tk in nocon_rep.keys():
                    tks[i] = nocon_rep[tk]
                elif tk == 'phone' and tks[i-1] == 'mobile':
                    tks[i-1] = 'cell'
                elif tk == 'engine' and 'vehicle' in tks:
                    tks[i-1] = 'locomotive'
                elif tk == 'mouse' and tks[i-1] != 'computer' and 'device' in tks:
                    tks[i] = 'computer mouse'
                elif tk == 'remote' and 'device' in tks and (i == len(tks)-1 or tks[i+1] != 'control'):
                    tks[i] = 'remote control'
                elif tk == 'Asian' and i < len(tks) - 1 and tks[i + 1] == 'food':
                    tks[i] = 'chinese'
                elif tk == 'machine' and tks[i-1] == 'coffee':
                    tks[i] = 'maker'
                elif tk == 'jeans':
                    try:
                        id = tks.index('made')
                        if id < i:
                            tks[i] = 'denim'
                    except:
                        try:
                            id = tks.index('make')
                            if id < i:
                                tks[i] = 'denim'
                        except:
                            try:
                                id = tks.index('makes')
                                if id < i:
                                    tks[i] = 'denim'
                            except:
                                pass
                if tks[i] == 'television' and i < len(tks) - 1 and tks[i + 1] == 'stand':
                    tks[i] = 'tv'
            que = ' '.join(tks)

            if 'tee' in tks and 'shirt' in tks:
                que = que.replace('tee shirt', 't-shirt')
            if 'wet' in tks and 'suit' in tks:
                que = que.replace('wet suit', 'wetsuit')
            if 'snow' in tks and 'suit' in tks:
                que = que.replace('snow suit', 'snowsuit')
            if 'jump' in tks and 'suit' in tks:
                que = que.replace('jump suit', 'jumpsuit')
            if 'swim' in tks and 'suit' in tks:
                que = que.replace('swim suit', 'swimsuit')
            if 'laptop' in tks and 'computer' in tks:
                que = que.replace('laptop computer', 'laptop')
            if 'dish' in tks and 'washer' in tks:
                que = que.replace('dish washer', 'dishwasher')
            if 'sea' in tks and 'gull' in tks:
                que = que.replace('sea gull', 'seagull')
            if 'cookie' in tks and 'sheet' in tks:
                que = que.replace('cookie sheet', 'baking sheet')
            if 'laptop' in tks and 'keyboard' in tks:
                que = que.replace('laptop keyboard', 'keyboard of the laptop')
            if 'laptop' in tks and 'screen' in tks:
                que = que.replace('laptop screen', 'screen of the laptop')
            if 'computer' in tks and 'screen' in tks:
                que = que.replace('computer screen', 'screen of the computer')
            if 'monitor' in tks and 'screen' in tks:
                que = que.replace('computer screen', 'screen of the monitor')
            if 'truck' in tks and 'taxi' in tks:
                que = que.replace('truck taxi', 'taxi')
            if 'highway' in tks and 'truck' in tks:
                que = que.replace('highway truck', 'truck')
            if 'man' in tks and 'glove' in tks:
                que = que.replace('man glove', 'glove of the man')
            if 'man' in tks and 'hat' in tks:
                que = que.replace('man hat', 'hat of the man')
            if 'man' in tks and 'shirt' in tks:
                que = que.replace('man shirt', 'shirt of the man')
            if 'boy' in tks and 'shirt' in tks:
                que = que.replace('boy shirt', 'shirt of the boy')
            if 'hot' in tks and 'dog' in tks and 'bun' in tks:
                que = que.replace('hot dog bun', 'bun of the hot dog')
            if 'you' in tks:
                que = que.replace('do you think ', '')
            if 'What' in tks and "'" in tks:
                que = que.replace("What ' s ", 'What is ')
            if 'mozzarella' in tks and 'cheese' in tks:
                que = que.replace('mozzarella cheese', 'mozzarella')
            if 'Which' in tks and 'direction' in tks:
                que = que.replace('Which direction', 'Which side')
            if 'of' in tks and 'man' in tks:
                que = que.replace('of man', 'of the man')

            Q[qid]['question'] = que

        # correct several wrong samples
        if 'train' in split:
            try:
                Q['05179746']['answer'] = 'panda bear'
            except:
                pass
        if 'val' in split:
            Q['18342440']['answer'] = 'panda bear'

        with open('../model/processed_data/' + split + '_questions_clean.json', 'w') as f:
            json.dump(Q, f)
