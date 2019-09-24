# Catch Transient Objects in a Few Shots

This repository combines two recent deep learning algorithms to do something cool. It shows how to learn transient objects like pedestrian in real time video from a self-driving car. It brings together Deep SORT[1][3], an object segmentation and tracking alogirhtm with the few-shot Neural Statistician[2][4] to create a generator for pedestrian clips. 


## Dependencies
Code has been tested on

* torch 1.2.0
* numpy 1.16.2
* sklearn 0.20.3
* cv2 (OpenCV) 4.1.0
* PIL 6.1.0
* tqdm 4.32.1


## Installation
```
cd /home/[MyNameHere]
git clone https://github.com/htso/Pedestrian_in_few_shots.git
cd Pedestrian_in_few_shots
git clone https://github.com/nwojke/deep_sort.git
```

## Generate data
I assume you've cloned this github to /home/[MyNameHere]/Pedestrian_in_few_shots

Step 1 : Download MOT16 data from 

         https://motchallenge.net/data/MOT16/

         Under heading "Download" in middle of page, click "Download all data (1.9 Gb)". 
         Put MOT16.zip in /home/[MyNameHere]/Pedestrian_in_few_shots/data,

         cd /home/[MyNameHere]/Pedestrian_in_few_shots/data
         unzip MOT16.zip

Step 2 : Download the pre-trained model and detection files as described in Deep SORT's README,

         https://drive.google.com/open?id=18fKzfqnqhqW3s9zwsCbnVJ5XF2JFeqMp

         You will get two folers : detections/, networks/. Place them in /home/[MyNameHere]/Pedestrian_in_few_shots/deep_sort

Step 3 : Pick a video and generate its hypotheses.txt file. I choose MOT16-03 in deep_sort/test/ folder for illustration; feel free to pick others.

         cd /home/[MyNameHere]/Pedestrian_in_few_shots/deep_sort

         python deep_sort_app.py \
              --sequence_dir=/home/[MyNameHere]/Pedestrian_in_few_shots/data/test/MOT16-03 \
              --detection_file=/home/[MyNameHere]/Pedestrian_in_few_shots/deep_sort/detections/MOT16_POI_test/MOT16-03.npy \
              --display=True \
              --output_file=/home/[MyNameHere]/Pedestrian_in_few_shots/hypotheses.txt \
              --min_confidence=0.8 \
              --min_detection_height=0 \
              --nms_max_overlap=1.0 \
              --max_cosine_distance=0.2         

         This will create a file 'hypotheses.txt' in /home/[MyNameHere]/Pedestrian_in_few_shots.

Step 4 : Run generate_MOT_dat.py,

         cd /home/[MyNameHere]/Pedestrian_in_few_shots

         python generate_MOT_data.py \
              --the_video=MOT16-03 \
              --detection_file=/home/[MyNameHere]/Pedestrian_in_few_shots/deep_sort/hypotheses.txt \
              --video_dir=/home/[MyNameHere]/Pedestrian_in_few_shots/data/test \
              --out_dir= /home/[MyNameHere]/Pedestrian_in_few_shots/data \
              --max_height=160 \
              --max_width=96 

The data to run Neural Stat are now in data/.

## Run Experiment

Edit the parameters in runMe.sh, then

```
cd /home/[MyNameHere]/Pedestrian_in_few_shots
bash runMe.sh
```

## Acknowledgements
Special thanks to https://github.com/conormdurkan for his implementation of the Neural Statistician paper from which this repo draws heavily on.


## References

[1] Wojke, N., Bewley, A., & Paulus, D. (2017, September). Simple online and realtime tracking with a deep association metric. In 2017 IEEE International Conference on Image Processing (ICIP) (pp. 3645-3649). IEEE.

[2] Edwards, H., & Storkey, A. (2016). Towards a neural statistician. arXiv preprint ![arXiv:1606.02185](https://arxiv.org/pdf/1606.02185.pdf)

[3] https://github.com/conormdurkan/neural-statistician

[4] https://github.com/nwojke/deep_sort.git

  