# mipnerf-360:Pytorch implementation of mipnerf-360

#### 注：更推荐去直接运行multinerf的开源代码，此代码可用于原理的理解(简洁易懂)，其效率和效果当时并不好

#### 欢迎将代码修改成更优的版本：

考察mipnerf360一周前的开源代码与本仓库的不同，发现有以下几点：(1)参数化和IPE部分，未在mipnerf360源码中找到论文中变换矩阵P矩阵的形式，而是直接用2的幂次进行编码；(2)蒸馏训练部分，我本身写的就和原论文相反，因为我认为细采样才应该点数更多，这一点有待商榷(3)正则化部分，应该区别不大，但是我的写了一个for循环，似乎可以去掉

但是，目前最本质的问题在于训练速度和训练psnr过低，上述trick感觉并不本质，目前一是要解决训练速度的问题，我写的pytorch架构没有写优化之类的环节，所以可能很慢，要尽可能去除循环等部分，同时考虑架构优化；二是要解决psnr训练不收敛的问题，这个原因我认为是整个过程中某个环节还是有物理问题，导致缺乏约束或者问题不适定，但是最近要结束实习了没时间检查细节了，如有同学想修改代码可以检查相关细节。

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

   实验数据包括[nerf_synthetic](https://drive.google.com/drive/folders/128yBriW1IG_3NJ5Rp7APSTZsJqdJdfc1),[nerf_llff_data](https://drive.google.com/drive/folders/128yBriW1IG_3NJ5Rp7APSTZsJqdJdfc1),[nerf_360](https://jonbarron.info/mipnerf360/)

   本实验中，下载数据集后放置在`data`目录下，分别为`data/nerf_synthetic`、`data/nerf_llff_data`或`data/nerf_360`
   
   可以采用如下命令下载`nerf_360`数据集：
   
   ```
   mkdir data
   cd data
   mkdir nerf_360
   cd nerf_360
   wget http://storage.googleapis.com/gresearch/refraw360/360_v2.zip
   unzip 360_v2.zip
   ```

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

​	相关结果保存在`log`目录下

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
