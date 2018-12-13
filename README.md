## 1. Overview
The project contains two parts. Part 1 is about the prediction of battle result between two pokemons, and part 2 is the image generation using Generative Adversarial Networks.<br>
## 2. Dependence (pip install)
<pre><code>Numpy<br>
Pandas<br/>
Tensorflow<br/>
cv2<br/>
matplotlib<br/>
seaborn<br/>
prettytable<br/>
itertools<br/>
time<br/>
scikit-learn<br/>
SciPy<br/></code></pre>
## 3. Part1
### 3.1 Introduction
First we conduct data visualization, feature engineering and elo-rating , then we try several classical machine learning models to predict the combat result. The models include logistical regression, SVM, KNN, Adaboost, MLP and so on. We plot each model’s confusion matrix and ROC curve.<br>
### 3.2 File
The file contains data and code, the data part contains combats.csv and pokemon.csv.<br> 
The code has five parts: data visualization, feature engineering, elo-rating and algorithms<br/>
### 3.3 Result
![](part1/Result.png)<br/>
## 4. Part 2
### 4.1 Introduction
We perform image generation by implementing Generative adversarial networks.
### 4.2 File
Part 2 contains two codes<br/>
Code_1 generates images of two classical pokemons called pikachu and squirtle using WGAN.<br/>
Code_2 generates images of all kinds of pokemons using 14000 pictures by implementing DCGAN.<br/>
### 4.3 Usage
#### Code_1
<pre><code>python resize.py<br/>
python rgb.py<br/>
python GAN.py<br/></code></pre>
#### Code_2
you can use our dataset to train the model by<br/>
<pre><code>python main.py --input_height 64--input_width 64 --output_height 64 --output_width 64 --data pokemon --crop --train --epoch 80 --input_fname_pattern "*.jpg"</code></pre>
Also, you can use your own dataset, making sure data folder contains your dataset or you can use<br/>
<pre><code>$ mkdir data/DATA_NAME</code></pre>
then, add you images data into DATA_NAME folder and use our model by
<pre><code>python main.py --input_height **--input_width ** --output_height ** --output_width ** --data DATA_NAME --crop --train --epoch 80 --input_fname_pattern "*.jpg"</code></pre>
### 4.4 Generated result
Pikachu(5000 epoches, 50 epoches per image)<br>

![](part2/results/Pikachu.gif)<br/>
Squirtle(5000 epoches, 50 epoches per image)<br>
![](part2/results/Squirtle.gif)<br/>
14000_pokemons(80 epoches, 1 epoch per image)<br>
![](part2/results/Pokemon_all.gif)<br/>

## 5 Credit
https://zhuanlan.zhihu.com/p/24767059<br/>
[Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks](https://arxiv.org/abs/1511.06434)<br/>
https://github.com/carpedm20/DCGAN-tensorflow<br/>
https://github.com/llSourcell/Pokemon_GAN
