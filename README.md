# SMC 2021 Challenge 3 Submission
 This repo contains our submission for [Smoky Mountains Data Challenge 2021 - Challenge 3](https://smc-datachallenge.ornl.gov/data-challenges-2021/), where we perform semantic segmentation on a cross-domain urban scenery dataset with simulated and live
 images.
 
Our model is based on [TransUNet](https://github.com/Beckschen/TransUNet) and has been adapted for use with the challenge's dataset.

## Instructions
1. Download and execute the [Git LFS installer](https://git-lfs.github.com/) (for cloning large files).
2. In Git Bash, run `git lfs install`.
3. Clone this repository (`git clone <link-to-repo>`).
4. Install required packages (`pip install -r requirements.txt`).
5. Navigate to the challenge_solution folder and run segment_images.py with the following arguments:
- `--source_dir` (path to image folder)
- `--target_dir` (path to output folder)
- `--batch_size` (optional, default=8)
- eg. `python segment_images.py --source_dir=path/to/source --target_dir=path/to/target --batch_size=16`

![mixed_sim_sample_2](https://user-images.githubusercontent.com/71860925/129434954-836f9ca9-89d9-4b3f-b0c1-4a7187cd6914.png)
![mixed_real_sample](https://user-images.githubusercontent.com/71860925/129434965-fd179c6c-9358-42e2-91cf-21ab5a0402ec.png)
