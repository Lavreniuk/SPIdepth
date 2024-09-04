# SPIdepth: Strengthened Pose Information for Self-supervised Monocular Depth Estimation

</a> <a href='https://arxiv.org/abs/2404.12501'><img src='https://img.shields.io/badge/Paper-Arxiv-red'></a>

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/spidepth-strengthened-pose-information-for/monocular-depth-estimation-on-kitti-eigen)](https://paperswithcode.com/sota/monocular-depth-estimation-on-kitti-eigen?p=spidepth-strengthened-pose-information-for)
<br>
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/spidepth-strengthened-pose-information-for/monocular-depth-estimation-on-kitti-eigen-1)](https://paperswithcode.com/sota/monocular-depth-estimation-on-kitti-eigen-1?p=spidepth-strengthened-pose-information-for)
<br>
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/spidepth-strengthened-pose-information-for/unsupervised-monocular-depth-estimation-on)](https://paperswithcode.com/sota/unsupervised-monocular-depth-estimation-on?p=spidepth-strengthened-pose-information-for)
<br>
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/spidepth-strengthened-pose-information-for/monocular-depth-estimation-on-make3d)](https://paperswithcode.com/sota/monocular-depth-estimation-on-make3d?p=spidepth-strengthened-pose-information-for)

## Training

To train on KITTI, run:

```bash
python train.py ./args_files/hisfog/kitti/cvnXt_H_320x1024.txt
```
For instructions on downloading the KITTI dataset, see [Monodepth2](https://github.com/nianticlabs/monodepth2)

To finetune on KITTI, run:

```bash
python ./finetune/train_ft_SQLdepth.py ./conf/cvnXt.txt ./finetune/txt_args/train/inc_kitti.txt
```

To train on CityScapes, run:

```bash
python train.py ./args_files/args_cityscapes_train.txt
```
To finetune on CityScapes, run:

```bash
python train.py ./args_files/args_cityscapes_finetune.txt
```

For preparing cityscapes dataset, please refer to SfMLearner's [prepare_train_data.py](https://github.com/tinghuiz/SfMLearner/blob/master/data/prepare_train_data.py) script.
We used the following command:

```bash
python prepare_train_data.py \
    --img_height 512 \
    --img_width 1024 \
    --dataset_dir <path_to_downloaded_cityscapes_data> \
    --dataset_name cityscapes \
    --dump_root <your_preprocessed_cityscapes_path> \
    --seq_length 3 \
    --num_threads 8
```

## Pretrained weights and evaluation

You can download weights for some pretrained models here:

* [KITTI]()
* [CityScapes]()

To evaluate a model on KITTI, run:

```bash
python evaluate_depth_config.py args_files/hisfog/kitti/cvnXt_H_320x1024.txt
```

Make sure you have first run `export_gt_depth.py` to extract ground truth files.

And to evaluate a model on Cityscapes, run:

```bash
python ./tools/evaluate_depth_cityscapes_config.py args_files/args_cvnXt_H_cityscapes_finetune_eval.txt
```

The ground truth depth files can be found at [HERE](https://storage.googleapis.com/niantic-lon-static/research/manydepth/gt_depths_cityscapes.zip),
Download this and unzip into `splits/cityscapes`.

## Inference with your own iamges

```bash
python test_simple_SQL_config.py ./conf/cvnXt.txt
```

## Citation
If you find this project useful for your research, please consider citing:
~~~
@misc{lavreniuk2024spidepthstrengthenedposeinformation,
      title={SPIdepth: Strengthened Pose Information for Self-supervised Monocular Depth Estimation}, 
      author={Mykola Lavreniuk},
      year={2024},
      eprint={2404.12501},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2404.12501}, 
}
~~~
## Acknowledgement
This project is built on top of [SQLdepth](https://github.com/hisfog/SfMNeXt-Impl), and we are grateful for their outstanding contributions.
