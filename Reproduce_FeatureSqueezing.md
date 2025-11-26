# Feature Squeezing: Detecting Adversarial Examples in Deep Neural Networks

![Feature Squeezing Detection Framework](https://xuweilin.org/publications/detection_framework.png)

```
@inproceedings{xu2018feature,
  title={{Feature Squeezing: Detecting Adversarial Examples in Deep Neural Networks}},
  author={Xu, Weilin and Evans, David and Qi, Yanjun},
  booktitle={Proceedings of the 2018 Network and Distributed Systems Security Symposium (NDSS)},
  year={2018}
}
```

The note was created to help reproduce the results of the [Feature Squeezing paper](https://arxiv.org/pdf/1704.01155.pdf). The code was developed on Python 2, but should be runnable on Python 3 with tiny modifications.

The following commands have been re-written to reproduce the results of a class project that has added boundary attack and Jpeg Compression as a new squeezing method to the authors origional results. 

## 1. Install dependencies.

```bash
pip install -r requirements_cpu.txt
```

If you are going to run the code on GPU, install this list instead:
```bash
pip install -r requirements_gpu.txt
```

## 2. Fetch submodules.
```bash
git submodule update --init --recursive
```

## 3. Download the pre-trained target models.
```bash
mkdir downloads; curl -sL https://github.com/mzweilin/EvadeML-Zoo/releases/download/v0.1/downloads.tar.gz | tar xzv -C downloads
```


## 4. Download the pre-generated adversary examples. You may skip this step and generate adversarial examples on your machine.
```bash
mkdir results
curl -sL https://github.com/mzweilin/EvadeML-Zoo/releases/download/v0.1/results_MNIST_100_317f6_carlini.tar.gz | tar xzv -C results
curl -sL https://github.com/mzweilin/EvadeML-Zoo/releases/download/v0.1/results_CIFAR-10_100_de671_densenet.tar.gz | tar xzv -C results
curl -sL https://github.com/mzweilin/EvadeML-Zoo/releases/download/v0.1/results_ImageNet_100_a2749_mobilenet.tar.gz | tar xzv -C results
```

## 5. Download the imagenet dataset. 
Visit this link: https://image-net.org/challenges/LSVRC/2012/2012-downloads.php make an account and download the validation set. 
Upload the dataset and unzip it. 
Then in datasets/imagenet.py, change this path:  self.img_folder = "/home/pi047867/EvadeML-Zoo/downloads/ILSVRC2012_img_val_labeled_caffe"
to match the path where the images are stored. 

## 6. Evaluate with the MNIST dataset.
```bash
# Evaluate the robust classification accuracy and the detection performance.
python main.py --dataset_name MNIST --model_name carlini  \
--attacks "\
fgsm?eps=0.3;\
bim?eps=0.3&eps_iter=0.06;\
carlinili?targeted=next&batch_size=1&max_iterations=1000&confidence=10;\
carlinili?targeted=ll&batch_size=1&max_iterations=1000&confidence=10;\
carlinil2?targeted=next&batch_size=100&max_iterations=1000&confidence=10;\
carlinil2?targeted=ll&batch_size=100&max_iterations=1000&confidence=10;\
carlinil0?targeted=next&batch_size=1&max_iterations=1000&confidence=10;\
carlinil0?targeted=ll&batch_size=1&max_iterations=1000&confidence=10;\
boundary_attack?init_num_trials=500&max_iterations=1000&delta=0.02;\
jsma?targeted=next;\
jsma?targeted=ll;" \
--robustness "none;\
FeatureSqueezing?squeezer=bit_depth_1;\
FeatureSqueezing?squeezers=jpeg_30;\
FeatureSqueezing?squeezers=jpeg_40;\
FeatureSqueezing?squeezers=jpeg_50;\
FeatureSqueezing?squeezers=jpeg_60;\
FeatureSqueezing?squeezers=jpeg_70;\
FeatureSqueezing?squeezer=median_filter_2_2;\
FeatureSqueezing?squeezer=median_filter_3_3;" \
--detection "FeatureSqueezing?squeezers=bit_depth_1&distance_measure=l1&fpr=0.05;\
FeatureSqueezing?squeezers=bit_depth_2&distance_measure=l1&fpr=0.05;\
FeatureSqueezing?squeezers=jpeg_30,median_filter_2_2&distance_measure=l1&fpr=0.05;\
FeatureSqueezing?squeezers=jpeg_30,bit_depth_1&distance_measure=l1&fpr=0.05;\
FeatureSqueezing?squeezers=bit_depth_1,median_filter_2_2&distance_measure=l1&fpr=0.05;"
```


## 7. Evaluate with the CIFAR-10 dataset.
```bash
python main.py --dataset_name CIFAR-10 --model_name DenseNet \
--attacks "fgsm?eps=0.0156;bim?eps=0.008&eps_iter=0.0012;carlinili?targeted=next&confidence=5;carlinili?targeted=ll&confidence=5;deepfool?overshoot=10;carlinil2?targeted=next&batch_size=100&max_iterations=1000&confidence=5;carlinil2?targeted=ll&batch_size=100&max_iterations=1000&confidence=5;carlinil0?targeted=next&confidence=5;carlinil0?targeted=ll&confidence=5;jsma?targeted=next;jsma?targeted=ll;boundary_attack?init_num_trials=500&max_iterations=1000&delta=0.02;" \
--robustness "none;\
FeatureSqueezing?squeezer=bit_depth_5;\
FeatureSqueezing?squeezer=bit_depth_4;\
FeatureSqueezing?squeezer=median_filter_2_2;\
FeatureSqueezing?squeezers=jpeg_30;\
FeatureSqueezing?squeezers=jpeg_40;\
FeatureSqueezing?squeezers=jpeg_50;\
FeatureSqueezing?squeezers=jpeg_60;\
FeatureSqueezing?squeezers=jpeg_70;\
FeatureSqueezing?squeezer=non_local_means_color_13_3_4;" \
--detection "\
FeatureSqueezing?squeezers=bit_depth_1&distance_measure=l1&fpr=0.05;\
FeatureSqueezing?squeezers=bit_depth_2&distance_measure=l1&fpr=0.05;\
FeatureSqueezing?squeezers=bit_depth_3&distance_measure=l1&fpr=0.05;\
FeatureSqueezing?squeezers=bit_depth_4&distance_measure=l1&fpr=0.05;\
FeatureSqueezing?squeezers=bit_depth_5&distance_measure=l1&fpr=0.05;\
FeatureSqueezing?squeezers=median_filter_2_2&distance_measure=l1&fpr=0.05;\
FeatureSqueezing?squeezers=median_filter_3_3&distance_measure=l1&fpr=0.05;\
FeatureSqueezing?squeezers=non_local_means_color_11_3_2&distance_measure=l1&fpr=0.05;\
FeatureSqueezing?squeezers=non_local_means_color_11_3_4&distance_measure=l1&fpr=0.05;\
FeatureSqueezing?squeezers=non_local_means_color_13_3_2&distance_measure=l1&fpr=0.05;\
FeatureSqueezing?squeezers=non_local_means_color_13_3_4&distance_measure=l1&fpr=0.05;\
FeatureSqueezing?squeezers=jpeg_70,bit_depth_4&distance_measure=l1&fpr=0.05;\
FeatureSqueezing?squeezers=jpeg_70,median_filter_2_2&distance_measure=l1&fpr=0.05;\
FeatureSqueezing?squeezers=jpeg_70,non_local_means_color_13_3_4&distance_measure=l1&fpr=0.05;\
FeatureSqueezing?squeezers=bit_depth_5,median_filter_2_2,non_local_means_color_13_3_2&distance_measure=l1&fpr=0.05;"
```


## 8. Evaluate with the ImageNet dataset.
```bash
python main.py --dataset_name ImageNet --model_name MobileNet --nb_examples 5 \
--attacks "fgsm?eps=0.0078;bim?eps=0.0040&eps_iter=0.0020;carlinili?batch_size=1&targeted=next&confidence=5;carlinili?batch_size=1&targeted=ll&confidence=5;deepfool?overshoot=35;carlinil2?max_iterations=1000&batch_size=10&targeted=next&confidence=5;carlinil2?max_iterations=1000&batch_size=50&targeted=ll&confidence=5;carlinil0?batch_size=1&targeted=next&confidence=5;carlinil0?batch_size=1&targeted=ll&confidence=5;boundary_attack?init_num_trials=500&max_iterations=1000&delta=0.02;" \
--robustness "none;\
FeatureSqueezing?squeezer=bit_depth_5;\
FeatureSqueezing?squeezer=bit_depth_4;\
FeatureSqueezing?squeezer=median_filter_2_2;\
FeatureSqueezing?squeezers=jpeg_30;\
FeatureSqueezing?squeezers=jpeg_40;\
FeatureSqueezing?squeezers=jpeg_50;\
FeatureSqueezing?squeezers=jpeg_60;\
FeatureSqueezing?squeezers=jpeg_70;\
FeatureSqueezing?squeezer=non_local_means_color_13_3_4;" \
--detection "\
FeatureSqueezing?squeezers=bit_depth_1&distance_measure=l1&fpr=0.05;\
FeatureSqueezing?squeezers=bit_depth_2&distance_measure=l1&fpr=0.05;\
FeatureSqueezing?squeezers=bit_depth_3&distance_measure=l1&fpr=0.05;\
FeatureSqueezing?squeezers=bit_depth_4&distance_measure=l1&fpr=0.05;\
FeatureSqueezing?squeezers=bit_depth_5&distance_measure=l1&fpr=0.05;\
FeatureSqueezing?squeezers=median_filter_2_2&distance_measure=l1&fpr=0.05;\
FeatureSqueezing?squeezers=median_filter_3_3&distance_measure=l1&fpr=0.05;\
FeatureSqueezing?squeezers=non_local_means_color_11_3_2&distance_measure=l1&fpr=0.05;\
FeatureSqueezing?squeezers=non_local_means_color_11_3_4&distance_measure=l1&fpr=0.05;\
FeatureSqueezing?squeezers=non_local_means_color_13_3_2&distance_measure=l1&fpr=0.05;\
FeatureSqueezing?squeezers=non_local_means_color_13_3_4&distance_measure=l1&fpr=0.05;\
FeatureSqueezing?squeezers=jpeg_70,bit_depth_3&distance_measure=l1&fpr=0.05;\
FeatureSqueezing?squeezers=bit_depth_5,median_filter_2_2,non_local_means_color_13_3_2&distance_measure=l1&fpr=0.05;"
```


