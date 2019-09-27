import numpy as np
import random
import sys
import os
import time
from skimage.io import imread
from torch.utils import data

# Directory structure 
#
#                                         TOP (data_dir)
#                                             |
#             -----------------------------------------------------------------
#             |                          |                                    |
#          Person1                    Person2          ....                PersonN
#             |                          |                                    |
#        ------------------        --------------------               ---------------------
#        |       |  ...   |        |       |  ...     |               |       |  ...      |  
#     video1  video2    videoM   video1  video2    videoM           video1  video2     videoM
#        |       |        |        |       |          |               |       |           |
#     frame1   frame1   frame1   frame1  frame1     frame1          frame1  frame1      frame1
#     frame2   frame2   frame2   frame2  frame2     frame2          frame2  frame2      frame2   
#       ...     ...      ...       ...    ...         ...             ...     ...         ...
#
#     frame_n  .....
#

class MOTDataset(data.Dataset):
    def __init__(self, data_dir, train_val_test_split, split, K, verbose=False):
        self.K = K
        self.path = data_dir
        self.verbose = verbose
        pedestrians = os.listdir(self.path) # these are folders, each associated with one pedestrian
        if verbose is True:
            print('split : ', split)
            print('pedestrians (folder) : ', pedestrians)

        #random.shuffle(pedestrians) # mix the order of the PERSON folders randomly
    
        n = len(pedestrians)
        n1 = int(n * train_val_test_split['train'])
        n2 = int(n * train_val_test_split['valid'])
        n3 = n - n1 - n2
        if verbose is True:
            print('total # pedestrians = %d   train=%d   val=%d  test=%d' %(n, n1, n2, n3))

        splits = {
            'train': slice(n1),
            'valid': slice(n1, n1+n2),
            'test': slice(n1+n2, n)
        }

        self.people = pedestrians[splits[split]] # [Person1, Persion2, ..., PersonN]
        if verbose is True:
            print('init self.people ', self.people)

        self.videos = [sorted(os.listdir(os.path.join(self.path, some_guy)))
                       for some_guy in self.people]
        # self.videos is a list of lists, each list is associated with a pedestrian, and each folder in that list
        # is a VIDEOn folder of video frames of that particular pedestrian  
        if verbose is True:            
            print('self.videos ', self.videos)                       
            # [[VIDEO1, VIDEO2], [VIDEO1, VIDEO2, VIDEO3], [...]]              
        
        for i in range(len(self.videos)):
            Videos = self.videos[i]
            # find the number of frames of all the videos 
            nfr = np.array([len(os.listdir(os.path.join(self.path, self.people[i], v))) for v in Videos])
            # which ones has at least K frames
            ii = list(np.where(nfr >= self.K)[0])
            # ... only select these videos
            Videos_at_least_K = [Videos[j] for j in ii]
            self.videos[i] = Videos_at_least_K

        one_guy = self.people[0]
        one_video = self.videos[0][0]

        if verbose is True:
            print('one guy ', one_guy)
            print('init video ', one_video)
        
        video_path = os.path.join(self.path, one_guy, one_video)
        all_frames = os.listdir(video_path)
        fr = all_frames[0]
        img = imread(os.path.join(video_path, fr))
        self.height = img.shape[0]
        self.width = img.shape[1]
        self.channels = img.shape[2]
        
        if verbose is True:
            print('sample image shape : ', img.shape)

        self.n = len(self.people)

    def __getitem__(self, ix):
        '''
        Each call selects a pedestrian (from self.people), supposedly in random. Then this function
        randomly chooses a video from all the videos associated with this pedestrian. 
        '''
        #print('getitem ix ', ix)
        pedestrian = self.people[ix]
        if self.verbose is True:
            print('getitem pedestrian ', pedestrian)

        # make sure the random choice below is different on each parallel thread (hack to solve multiprocessing rng issue)
        seed = int(str(time.time()).split('.')[1])
        np.random.seed(seed=seed)
        
        # randomly choose a video from all the videos on this pedestrian
        Videos = self.videos[ix]
        if self.verbose is True: 
            print('Videos type :', type(Videos))
            print('len : ', len(Videos))
            print(Videos)

        # find the number of frames of all the videos 
        # nfr = np.array([len(os.listdir(os.path.join(self.path, pedestrian, v))) for v in Videos])
        # which ones has at least K frames
        # ii = list(np.where(nfr >= self.K)[0])
        # print('ii type :', type(ii))
        # print('ii ', ii)
        # if len(ii) == 0:
        #     print("Something's wrong. nfr : ", nfr )
        #     print("pedestrian : ", pedestrian)
        #     print('Videos :', Videos )
        #     sys.exit()

        # # ... only select these videos
        # Videos_at_least_K = [Videos[j] for j in ii] 

        video_picked = np.random.choice(Videos)
        video_path = os.path.join(self.path, pedestrian, video_picked)
        all_frames = os.listdir(video_path)

        if self.verbose is True:
            print('Vidoes_at_least_K :', Vidoes_at_least_K)
            print('video picked :', video_picked)
            print('video_path : ', video_path)
            print('# of frames in this video : ', len(all_frames))

        # m = 0
        # while len(all_frames) < self.K and m < 5 :
        #     video_picked = np.random.choice(Videos)
        #     video_path = os.path.join(self.path, pedestrian, video_picked)
        #     all_frames = os.listdir(video_path)

        #     if self.verbose is True:
        #         print('video picked :', video_picked)
        #         print('# of frames in this video : ', len(all_frames))
        #         print('m : ', m)

        #     m = m + 1
        
        # if m >= 5 and len(all_frames) < self.K:
        #     print("something's wrong. couldn't find a video with at least K frames after 5 trials. len(all_frames) = %d" % (len(all_frames)))
        #     print('pedestrian :', pedestrian)
        #     print('Videos :', Videos )
        #     sys.exit()

        # choose frames from video
        choice_frames = np.random.choice(all_frames, size=self.K, replace=False)
        
        return np.array([imread(os.path.join(video_path, frame)).transpose(2, 0, 1)
                         for frame in choice_frames]).astype(np.float32) / 255

    def __len__(self):
        return self.n


