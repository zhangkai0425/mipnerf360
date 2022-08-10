# mipnerf-360:Pytorch implementation of mipnerf-360

#### 运行准备：

1. 克隆代码到本地仓库

```shell
git clone https://github.com/zhangkai0425/mipnerf-360.git
```

2. 配置相应环境   `cuda:11.0|NVIDIA GeForce RTX 3090` 

从头开始搭建环境，推荐：

```shell
cd mipnerf-360
cd env
conda create -n mipnerf360 python=3.7
conda activate mipnerf360
pip install -r requirements.txt
```

或者

```shell
cd mipnerf-360
cd env
conda env create -f environment.yaml
conda activate mip-NeRF
```

3. 数据集准备

   实验数据包括[nerf_synthetic](),[nerf_llff_data](),[nerf_360]()

   本实验中，下载数据集后放置在`data`目录下，分别为`data/nerf_synthetic`、`data/nerf_llff_data`或`data/nerf_360`

4. 设置参数

   所有的参数均在 `config.py`  中，可在命令行运行时设置覆盖默认参数

#### 训练

1. 从零开始训练Lego场景(不推荐训练`nerf_synthetic`类数据)

```shell
cd mipnerf-360
# train nerf_synthetic/lego data
python train.py --log_dir log/lego --dataset_name blender --scene lego
```

2. 从零开始训练LLFF场景

```shell
cd mipnerf-360
# train llff data:change the scene_name to the scene to be trained
python train.py --log_dir log/scene_name --dataset_name llff --scene scene_name
```

3. 从零开始训练nerf_360场景

```shell
cd mipnerf-360
# train nerf_360 data:scene garden
python train.py --log_dir log/garden --dataset_name nerf_360 --scene garden
```

#### 测试

测试`nerf-360`数据下的模型，其他类似

1. 进入根目录下

```shell
cd mipnerf-360
```

2. 运行`test.py`

```shell
python test.py --log_dir log/garden --dataset_name nerf_360 --scene garden --model_weight_path log/garden/model_10000.pt --visualize_depth --visualize_normals
```

​	相关结果保存在`log/test`目录下

#### 可视化

可视化`nerf-360`数据下的模型，生成视频，其他类似

1. 进入根目录下

```shell
cd mipnerf-360
```

2. 运行`visualize.py`

```shell
python video.py --log_dir log/garden --dataset_name nerf_360 --scene garden --model_weight_path log/garden/model_10000.pt --visualize_depth --visualize_normals
```

​	相关结果保存在`log/test`目录下

#### demo

在`demo`文件夹下包含相应的`scripts`脚本样式，给出了一般形式的训练、测试、可视化命令格式，可单独复制运行，或将`.sh`文件置于主文件夹目录下运行

#### 文件清单

```bash
...
+ demo              	        # demo scripts
+ env              		# environment
	-- requirements.txt
	-- environment.yaml
+ intern            	        # all the keypoints in mipnerf360
	-- distillation.py
	-- encoding.py
	-- loss.py
	-- parameterization.py
	-- pose.py
	-- ray.py
	-- regularization.py
	-- scheduler.py
	-- utils.py
+ log                           # log directory
README.md                       # README
config.py            	        # config
dataset.py                      # dataset loader
model.py            	        # mipnerf360 model
test.py              	        # test the model:generate the target images
train.py             	        # train the model
video.py                        # visualize and generate the videos
```
