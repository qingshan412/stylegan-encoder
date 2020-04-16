#!/bin/bash

#$ -M jliu16@nd.edu      # Email address for job notification
#$ -m abe                # Send mail when job begins, ends and aborts
#$ -pe smp 16            # Specify parallel environment and legal core size
#$ -q gpu
#$ -l gpu_card=1
#$ -N z_encode_gpu_p         # Specify job name

module load tensorflow/1.12

python encode_images_p.py cache/dist/st_1024 cache/dist/rcs_256_st_1024_b10_p cache/dist/latent_256_st_1024_b10_p --batch_size=10
# python encode_images_p.py cache/dist/st_1024 cache/dist/rcs_128_st_1024_p cache/dist/latent_128_st_1024_p --image_size=128
# python encode_images_p.py cache/dist/st_1024 cache/dist/rcs_112_st_1024_p cache/dist/latent_112_st_1024_p --image_size=112
# python encode_images_p.py cache/dist/st_1024 cache/dist/rcs_256_st_1024_p cache/dist/latent_256_st_1024_p
