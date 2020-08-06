# TensorflowKeras Implementation of Single Shot MultiBox Detector
A pure Tensorflow+Keras Implementation of [SSD (Single Shot MultiBox Detector)](https://arxiv.org/abs/1512.02325) using different backbones of Efficientnet on the PASCAL_VOC dataset.

![Example of EfficientnetB3 SSD](example.jpg  "Example of EfficientnetB3 SSD")

## Dependencies
1. Python 3.6+
2. Tensorflow 2.2.0+
3. Tensorflow_Datasets 3.0.0+
4. Efficientnet
5. Keras 2.4.0+ (also called 2.3.0-tf)

To install these dependencies, run
```bash
pip install -r requirements.txt
```
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

### EfficientNetB3 SSD (300 x 300)

|     Class    | Average Precision |
| :----------: |   :----------:    |
| aeroplane | 0.8138464069658522  |
| bicycle | 0.8495817279095099  |
| bird | 0.7411210589421086  |
| boat | 0.6953653156671167  |
| bottle | 0.44501229699670486  |
| bus | 0.8169842240921157  |
| car | 0.8348353061776034  |
| cat | 0.867478526227363  |
| chair | 0.6359432618499943  |
| cow | 0.7347537326752558  |
| diningtable | 0.7556188461308526  |
| dog | 0.8453690345281261  |
| horse | 0.8431191113749101  |
| motorbike | 0.8414147988740155  |
| person | 0.8000506418143006  |
| pottedplant | 0.5718438091150522  |
| sheep | 0.7248609390832621  |
| sofa | 0.8101149566442967  |
| train | 0.8836348165299509  |
| tvmonitor | 0.7444889524463032  |


#### Mean Average Precision: 0.7627718882022347


## References
1. https://github.com/qfgaohao/pytorch-ssd
2. https://github.com/lufficc/SSD
3. https://github.com/mvoelk/ssd_detectors