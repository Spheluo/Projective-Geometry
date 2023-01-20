# Projective-Geometry
homography estimation, forward and backward warping, marker-based planar AR, Panorama

## part 1. Homography Estimation
### Implements forward warping to warp images into canvas with part1.py. The DLT method is based on singular value decomposition (SVD). Familiar DLT (direct linear transform) estimation method is as follows:
<img width="400" alt="截圖 2023-01-20 下午12 14 22" src="https://user-images.githubusercontent.com/96567794/213615998-3b085a48-9239-4676-9b5d-5ac918b3d723.png"> <img width="400" alt="times" src="https://user-images.githubusercontent.com/96567794/213615721-0bb223b9-eb60-4911-b303-72ba33797aa5.jpg"> <img width="400" alt="截圖 2023-01-20 下午12 14 57" src="https://user-images.githubusercontent.com/96567794/213616065-a00701bc-9ae9-48cc-a775-aa944309d828.png"> <img width="400" alt="output1" src="https://user-images.githubusercontent.com/96567794/213615416-31d9efe6-dd9a-4601-b124-22ad8a395ba1.png">


$ python3 part1.py

## part 2. Marker-Based Planar AR
### Familiar with off-the-shelf ArUco marker detection tool and practice backward warping with part2.py
$ python3 part2.py

## part 3. Unwarp the secret
### What can go wrong with practical homography?
<img width="1075" alt="截圖 2023-01-20 下午12 04 30" src="https://user-images.githubusercontent.com/96567794/213615065-3cb6c723-5dc3-4d0f-ae5f-556a432dfba7.png">
$ python3 part3.py

## part 4. Panorama
### Practice panorama stitching with ORB feature detector, brute-force matching with ORB Descriptors, and RANSAC for selecting best homography.
![output4](https://user-images.githubusercontent.com/96567794/213615123-c89e3c29-bf8d-430f-bd8d-1945b0da619d.png)
$ python3 part4.py
