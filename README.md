# TensorflowKeras Implementation of Single Shot MultiBox Detector
A pure Tensorflow+Keras Implementation of [SSD (Single Shot MultiBox Detector)](https://arxiv.org/abs/1512.02325) using different backbones of Efficientnet on the PASCAL_VOC dataset.

![Example of EfficientnetB3 SSD](example.jpg  "Example of EfficientnetB3 SSD")

## Dependencies
1. Python 3.6+
2. Tensorflow 2.2.0+
3. Tensorflow_Datasets 3.0.0+
4. Efficientnet
5. Keras 2.4.0+ (also called 2.3.0-tf)

## Test on your own images
Add your images to the `inputs/` folder and then run
```bash
python predict.py
```
A pretrained model with EfficientNetB3 backbone will load and run on all images in the `inputs/` folder. The results can be found in the `outputs/` folder.

## NOTE: 
To train or evaluate the model would require downloading the PASCAL VOC dataset and converting it into tfrecords format by `tensorflow_datasets` module. This is done automatically in `train.py` or `eval.py` but requires a considerable amount of time on the first run.

## Evaluate Model
In `eval.py`, change the `checkpoint_filepath` variable to your trained model weights and make sure the base model i.e. `MODEL_NAME` is set according to your architecture ('B0', 'B1'...etc.). Then run
```bash
python eval.py
```
PASCAL_VOC evaluation will be performed on the VOC2007 test dataset.

## Train Model
To train from scratch, run
```bash
python train.py
```
If you want to continue training from a checkpoint set `checkpoint_filepath` and set `base_lr` accordingly.

## Pretrained Models

### EfficientNetB3 SSD

```
 Class     Average_Precision

'aeroplane': 0.7922828141612303
'bicycle': 0.8474897444195838 
'bird': 0.6786566002663461
'boat': 0.6824745082560313
'bottle': 0.34892944871248044
'bus': 0.8062074367653329 
'car': 0.8214033679011029
'cat': 0.8692819157138691 
'chair': 0.601266933531285 
'cow': 0.6818036339967353  
'diningtable': 0.7478443406289889
'dog': 0.8485163395720235
'horse': 0.8432749067332937
'motorbike': 0.8321709286868102
'person': 0.7577312302797639
'pottedplant': 0.5281010714446207
'sheep': 0.709128135610284
'sofa': 0.8273209136585827
'train': 0.909418618911262
'tvmonitor': 0.731836624269091

Mean Average Precision: 0.7432569756759358
```


## References
1. https://github.com/qfgaohao/pytorch-ssd
2. https://github.com/lufficc/SSD
3. https://github.com/mvoelk/ssd_detectors