# SMC 2021 Challenge 3 Submission
 This repo contains our submission for Smoky Mountains Data Challenge 2021 - Challenge 3, where we perform semantic segmentation on a cross-domain dataset with simulated and live
 images.
 
Our model is based on TransUNet (https://github.com/Beckschen/TransUNet) and has been adapted for use with the challenge's dataset

## Instructions
1. Ensure required packages are installed (requirements.txt)
2. Navigate to the challenge_solution folder and run segment_images.py with the following arguments:
- --source_dir (path to image folder)
- --target_dir (path to output folder)
- --batch_size (optional, default=8)

![mixed_sim_sample_2](https://user-images.githubusercontent.com/71860925/129434954-836f9ca9-89d9-4b3f-b0c1-4a7187cd6914.png)
![mixed_real_sample](https://user-images.githubusercontent.com/71860925/129434965-fd179c6c-9358-42e2-91cf-21ab5a0402ec.png)
