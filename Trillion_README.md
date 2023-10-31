1. Install GPU driver

   ```
   https://www.nvidia.com/download/index.aspx
   ```

   

2. Install CUDA toolkit

   ```
   sudo apt install nvidia-cuda-toolkit
   ```

3. Clone project

   ```
   cd YOUR_CODE_PATH
   git clone https://github.com/nv-tlabs/GET3D.git
   cd GET3D; mkdir cache; cd cache
   wget https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan3/versions/1/files/metrics/inception-2015-12-05.pkl
   ```

4. Download pretrained models from [here.](https://drive.google.com/drive/folders/1oJ-FmyVYjIwBZKDAQ4N1EEcE9dJjumdW?usp=sharing)

   Add folder get3d_release to YOUR_CODE_PATH

5. Install nvidia-container-runtime

   ```
   https://docs.docker.com/config/containers/resource_constraints/#gpu
   ```

6. Build docker image

   ```
   cd docker
   chmod +x make_image.sh
   ./make_image.sh get3d:v1
   ```

7. Run docker image (from YOUR_CODE_PATH)

   ```
   docker run --gpus device=all -it --rm -v ${PWD}:/workspace/GET3D -it get3d:v1 bash
   export PYTHONPATH=$PWD:$PYTHONPATH
   export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
   ```

8. Inference (from YOUR_CODE_PATH)

   ```
   python train_3d.py --outdir=save_inference_results/shapenet_car  --gpus=1 --batch=4 --gamma=40 --data_camera_mode shapenet_car  --dmtet_scale 1.0  --use_shapenet_split 1  --one_3d_generator 1  --fp32 0 --inference_vis 1 --resume_pretrain get3d_release/shapenet_car.pt
   ```

   ```
   python train_3d.py --outdir=save_inference_results/shapenet_motorbike  --gpus=1 --batch=4 --gamma=40 --data_camera_mode shapenet_motorbike  --dmtet_scale 1.0  --use_shapenet_split 1  --one_3d_generator 1  --fp32 0 --inference_vis 1 --resume_pretrain /content/GET3D/get3d_release/shapenet_motorbike.pt
   ```

9. Train 