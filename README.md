# Magiclayouts
Code and models for Magic Layouts (CVPR 2021)


## Dataset
### RICO Dataset from [rico](https://interactionmining.org/rico)
### DrawnUI HandDrawn sketches of UX designs (Please contact if you need a copy of this)

## To do the inference 
* This repo depends on  pytorch>1.4 and detectron2. PLease install them first.
* First Download the pretrained models from [here](https://drive.google.com/drive/folders/1C28cQ3oZXwJ9oOJS0lZRpDLIfDaXSanx?usp=sharing) and put them in `trained_models`
* Run predict.py
* You can choose the dataset/model in the script

If you find this repository useful for your publications, please consider citing our paper.

```
@inproceedings{manandhar2021magic,
  title={Magic Layouts: Structural Prior for Component Detection in User Interface Designs},
  author={Manandhar, Dipu and Jin, Hailin and Collomosse, John},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={15809--15818},
  year={2021}
}
```

