#!/bin/bash

#$ -M jliu16@nd.edu      # Email address for job notification
#$ -m abe                # Send mail when job begins, ends and aborts
#$ -pe smp 16            # Specify parallel environment and legal core size
#$ -q gpu
#$ -l gpu_card=1
#$ -N encode_test_gpu         # Specify job name

module load tensorflow/1.12

python encode_images.py cache/dist/st_1024 cache/dist/rcs_1024_st_1024 cache/dist/latent_st_1024_1024 --image_size=1024
python encode_images.py cache/dist/st_1024 cache/dist/rcs_256_st_1024_b10 cache/dist/latent_st_1024_b10 --batch_size=10
# python encode_images.py cache/dist/st_1024 cache/dist/rcs_256_st_1024 cache/dist/latent_st_1024
# python align_images.py cache/dist/orig cache/dist/st_1024
# st: set (size), rcs: reconstructed, 