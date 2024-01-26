import json
import numpy as np
from tqdm import tqdm

if __name__ == '__main__':
    #train = json.load(open('model/processed_data/train_balanced_questions_clean.json'))
    #val = json.load(open('model/processed_data/val_balanced_questions_clean.json'))
    #test = json.load(open('model/processed_data/testdev_balanced_questions_clean.json'))
    stat = ['does', 'do', 'is', 'are']#, 'was', 'did', 'were', 'could']
    foods1 = ['sauce', 'cake', 'donut', 'icing', 'frosting', 'cupcake', 'milkshake', 'syrup']
    foods2 = ['ice cream']
    ans2idx = json.load(open('../model/processed_data/ans2idx_all.json'))
    all_colors = ['dark', 'brown', 'blond', 'blue', 'green', 'yellow', 'black', 'gray', 'white', 'red', 'purple',
                  'light brown', 'light blue', 'silver', 'pink', 'orange', 'tan', 'dark brown', 'beige', 'gold',
                  'cream colored', 'maroon', 'dark blue', 'khaki', 'teal', 'brunette']
    color_ids = [ans2idx[i] for i in all_colors]
    ball = ['baseball', 'tennis', 'soccer']
    yes_id, no_id = ans2idx['yes'], ans2idx['no']
    all_directions = ['left', 'right']
    left_or_right_ids = [ans2idx[i] for i in all_directions]
    all_sizes = ['large', 'small', 'huge', 'little', 'giant', 'tiny']
    size_ids = [ans2idx[i] for i in all_sizes]
    all_ages = ['little', 'young', 'old']
    age_ids = [ans2idx[i] for i in all_ages]
    all_lengths = ['long', 'short']
    length_ids = [ans2idx[i] for i in all_lengths]
    all_heights = ['tall', 'short']
    height_ids = [ans2idx[i] for i in all_heights]
    
    yes_or_no = dict()
    alternative = dict()
    color = dict()
    direction = dict()
    size = dict()
    age = dict()
    length = dict()
    height = dict()

    for split in ['testdev_all']:#['train_balanced', 'val_balanced', 'testdev_balanced']:#['submission_all', 'test_all']:
        Q = json.load(open('../model/processed_data/' + split + '_questions_clean.json'))
        ans_mask = dict()
        que_type = dict()
        for key in ['gt', 'pd', 'hit']:
            yes_or_no[key] = 0
            alternative[key] = 0
            color[key] = 0
            direction[key] = 0
            size[key] = 0
            age[key] = 0
            length[key] = 0
            height[key] = 0

        yes_or_no['yes'] = 0
        yes_or_no['no'] = 0

        for qid in tqdm(Q.keys()):
            que_type[qid] = 0
            ans_mask[qid] = []
            tks = Q[qid]['question'].lower().split()  # delete '?' in the end
            lower_que = (Q[qid]['question'].lower())
            if Q[qid]['answer'] == 'yes':
                yes_or_no['gt'] += 1
                yes_or_no['yes'] += 1
            elif Q[qid]['answer'] == 'no':
                yes_or_no['gt'] += 1
                yes_or_no['no'] += 1

            if (tks[0] in stat) and (tks[1] == 'there' or 'or' not in tks or (tks[1] == 'you' and 'any' in tks) or (tks[1] == 'you' and tks[2] == 'see')):  #45842
                # yes or no
                que_type[qid] = 1
                ans_mask[qid].extend([yes_id, no_id])
                yes_or_no['pd'] += 1
                if Q[qid]['answer'] in ['yes', 'no']:
                    yes_or_no['hit'] += 1
            elif 'or' in tks:
                # choose Q
                que_type[qid] = 2
                or_pos = tks.index('or')
                left_stat = or_pos - 1
                right_stat = or_pos + 1
                if tks[right_stat] == 'to':
                    right_stat += 1
                if tks[right_stat] == 'on' and right_stat < len(tks)-2:
                    right_stat += 1
                if tks[right_stat] == 'the':
                    right_stat += 1
                if tks[right_stat] == 'a':
                    right_stat += 1
                if tks[right_stat] == 'an':
                    right_stat += 1

                cand = []
                l_scores = np.zeros(len(ans2idx))
                l_found = False
                for i in range(left_stat, -1, -1):
                    for k, name in enumerate(ans2idx.keys()):
                        name_tks = name.split()
                        if len(name_tks) <= or_pos - i and sum([tks[i+j] != name_tks[j] for j in range(len(name_tks))]) == 0:
                            if l_scores[k] == 0:
                                l_scores[k] = (i + len(name_tks)) * 100 + len(name_tks) + 10000
                                if tks[i+len(name_tks)] == 'of' and tks[i+len(name_tks)+1] == 'the' and name != 'picture':
                                    l_scores[k] += 300 + 5
                                    if ' '.join([tks[i+len(name_tks)+2], tks[i+len(name_tks)+3]]) in ans2idx.keys():
                                        l_scores[k] += 100
                            l_found = True
                if l_found:
                    lid = int(l_scores.argmax())
                    ans_mask[qid].append(lid)
                    cand.append(list(ans2idx.keys())[lid])

                r_scores = np.zeros(len(ans2idx))
                r_found = False
                for i in range(right_stat, len(tks)+1):
                    for k, name in enumerate(ans2idx.keys()):
                        name_tks = name.split()
                        if len(name_tks) <= len(tks) - i and sum([tks[i+j] != name_tks[j] for j in range(len(name_tks))]) == 0:
                            if r_scores[k] == 0:
                                r_scores[k] = (-i * 100 + len(name_tks)) + 10000
                                if tks[i-1] + ' ' + tks[i] not in ans2idx.keys() and (len(cand) == 0 or cand[0] not in ball):
                                    if tks[i-2] + ' ' + tks[i-1] in ans2idx.keys():
                                        r_scores[k] += 202
                                    elif tks[i-1] in ans2idx.keys():
                                        r_scores[k] += 101
                                if len(name_tks) <= len(tks) - 1 - i and tks[i + len(name_tks)] in foods1:
                                    r_scores[k] += 105
                                elif len(name_tks) <= len(tks) - 2 - i and ' '.join([tks[i + len(name_tks)], tks[i + len(name_tks) + 1]]) in foods2:
                                    r_scores[k] += 205
                            r_found = True
                if r_found:
                    rid = int(r_scores.argmax())
                    if l_found and lid == ans2idx['indoors']:
                        rid = ans2idx['outdoors']
                    if l_found and lid == ans2idx['outdoors']:
                        rid = ans2idx['indoors']
                    ans_mask[qid].append(rid)
                    cand.append(list(ans2idx.keys())[rid])
                if not (r_found or l_found) or Q[qid]['answer'] not in cand:
                    alternative['pd'] += 1
                    print('Unextracted correct answer for Alternative', qid, Q[qid]['question'], Q[qid]['answer'])
                else:
                    alternative['pd'] += 1
                    alternative['hit'] += 1
            elif 'which color' in lower_que or 'what color' in lower_que or 'what is the color ' in lower_que:
                que_type[qid] = 3
                ans_mask[qid] = color_ids
                color['pd'] += 1
                if Q[qid]['answer'] in all_colors:
                    color['hit'] += 1
                    color['gt'] += 1
            elif 'which side ' in lower_que or 'which part ' in lower_que or 'which direction ' in lower_que \
                    or (tks[0] == 'where' and tks[-2] == 'facing') or (tks[0] == 'where' and tks[-3] == 'looking' and tks[-2] == 'at'):
                que_type[qid] = 4
                ans_mask[qid] = left_or_right_ids
                direction['pd'] += 1
                if Q[qid]['answer'] in all_directions:
                    direction['hit'] += 1
                    direction['gt'] += 1
                else:
                    print("Wrong direction: ", Q[qid]['question'], Q[qid]['answer'])
            elif 'how big ' in lower_que or 'how large ' in lower_que or 'what size ' in lower_que or 'what is the size ' in lower_que:
                que_type[qid] = 5
                ans_mask[qid] = size_ids
                size['pd'] += 1
                if Q[qid]['answer'] in all_sizes:
                    size['hit'] += 1
                    size['gt'] += 1
            elif 'how old ' in lower_que:
                que_type[qid] = 6
                ans_mask[qid] = age_ids
                age['pd'] += 1
                if Q[qid]['answer'] in all_ages:
                    age['hit'] += 1
                    age['gt'] += 1
            elif 'how long ' in lower_que or 'what is the length ' in lower_que or 'what length ' in lower_que:
                que_type[qid] = 7
                ans_mask[qid] = length_ids
                length['pd'] += 1
                if Q[qid]['answer'] in all_lengths:
                    length['hit'] += 1
                    length['gt'] += 1
            elif 'how tall ' in lower_que or 'what height ' in lower_que or 'what is the height ' in lower_que:
                que_type[qid] = 8
                ans_mask[qid] = height_ids
                height['pd'] += 1
                if Q[qid]['answer'] in all_heights:
                    height['hit'] += 1
                    height['gt'] += 1
            elif Q[qid]['answer'] in ['yes', 'no']:
                print("Miss YesOrNO", qid, Q[qid]['question'], Q[qid]['answer'])
            elif Q[qid]['answer'] in all_colors and Q[qid]['answer'] != 'orange':
                color['gt'] += 1
                print("Miss Color", qid, Q[qid]['question'], Q[qid]['answer'])
            elif Q[qid]['answer'] in all_directions:
                direction['gt'] += 1
                print("Miss Direction", qid, Q[qid]['question'], Q[qid]['answer'])
            elif Q[qid]['answer'] in all_sizes:
                size['gt'] += 1
                print("Miss Size", qid, Q[qid]['question'], Q[qid]['answer'])
            elif Q[qid]['answer'] in all_ages:
                age['gt'] += 1
                print("Miss Age", qid, Q[qid]['question'], Q[qid]['answer'])
            elif Q[qid]['answer'] in all_lengths:
                length['gt'] += 1
                print("Miss Length", qid, Q[qid]['question'], Q[qid]['answer'])
            elif Q[qid]['answer'] in all_heights:
                height['gt'] += 1
                print("Miss Height", qid, Q[qid]['question'], Q[qid]['answer'])

        print('{} Yes:{} No:{} Pre:{} Rec:{} Num:{}/{}'.format(split, yes_or_no['yes']/yes_or_no['gt'],
                                                               yes_or_no['no']/yes_or_no['gt'],
                                                               yes_or_no['hit']/yes_or_no['pd'],
                                                               yes_or_no['hit']/yes_or_no['gt'],
                                                               yes_or_no['gt'], len(Q)))
        print('{} Alternative Rec:{} Num:{}/{}'.format(split, alternative['hit'] / alternative['pd'], alternative['pd'], len(Q)))
        print('{} Color Pre:{} Rec:{} Num:{}/{}'.format(split, color['hit'] / color['pd'], color['hit'] / color['gt'],
                                                               color['gt'], len(Q)))
        print('{} Direction Pre:{} Rec:{} Num:{}/{}'.format(split, direction['hit'] / direction['pd'],
                                                            direction['hit'] / direction['gt'],
                                                            direction['gt'], len(Q)))
        print('{} Size Pre:{} Rec:{} Num:{}/{}'.format(split, size['hit'] / size['pd'], size['hit'] / size['gt'],
                                                       size['gt'], len(Q)))
        print('{} Age Pre:{} Rec:{} Num:{}/{}'.format(split, age['hit'] / age['pd'], age['hit'] / age['gt'],
                                                       age['gt'], len(Q)))
        print('{} Length Pre:{} Rec:{} Num:{}/{}'.format(split, length['hit'] / length['pd'], length['hit'] / length['gt'],
                                                         length['gt'], len(Q)))
        print('{} Height Pre:{} Rec:{} Num:{}/{}'.format(split, height['hit'] / height['pd'], height['hit'] / height['gt'],
                                                         height['gt'], len(Q)))
        with open('../model/processed_data/answer_mask_{}.json'.format(split), 'w') as f:
            json.dump(ans_mask, f)
        with open('../model/processed_data/question_type_{}.json'.format(split), 'w') as f:
            json.dump(que_type, f)
