import os
import os.path as osp
import collections


data_dir = 'data'
new_data_dir = 'data_sorted'

files = os.listdir(data_dir)

for file in files:
    dataset = file.split('.')[0]
    fh = open(osp.join(data_dir, file), 'r')
    lines = fh.readlines()

    print(dataset)
    if dataset in ['caltech', 'cuhksysu']:
        print('b')
        if dataset == 'caltech':
            print('a')

            sets = collections.defaultdict(lambda: {})
            for line in lines:
                name = line.split('/')[-1][:-5]
                set_ind = int(name[3:5])
                video_ind = int(name[7:10])
                index = int(name.split('_')[-1])

                if video_ind not in sets[set_ind]:
                    sets[set_ind][video_ind] = []

                sets[set_ind][video_ind].append(index)


            for set_ind in sets.keys():
                for video_ind in sets[set_ind].keys():
                    sets[set_ind][video_ind] = sorted(sets[set_ind][video_ind])



            # import pdb; pdb.set_trace()


            print(sets)


