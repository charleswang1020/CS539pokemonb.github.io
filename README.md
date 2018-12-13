## 1. Overview
The project contains two parts. Machine Learning part is about the prediction of battle result between two pokemons, and Deep Learning part is the image generation using Generative Adversarial Networks.<br>
## 2. Dependence (pip install)
<pre><code>Numpy<br>
Pandas<br/>
Tensorflow<br/>
cv2<br/>
matplotlib<br/>
seaborn<br/>
prettytable<br/>
itertools<br/>
scikit-learn<br/>
SciPy<br/></code></pre>
## 3. Machine Learning
### 3.1 Introduction
First we conduct data visualization, feature engineering and elo-rating , then we try several classical machine learning models to predict the combat result. The models include logistical regression, SVM, KNN, Adaboost, MLP and so on. We plot each model’s confusion matrix and ROC curve.<br>
### 3.2 File
The file contains data and code, the data part contains combats.csv and pokemon.csv.<br> 
The code has five parts: data visualization, feature engineering, elo-rating and algorithms<br/>
### 3.3 Result
The part1/code.ipynb includes all results of our machine learning part.<br/>
![](part1/Result.png)<br/>
## 4. Deep Learning
### 4.1 Introduction
We perform image generation by implementing Generative adversarial networks.
### 4.2 File
Code_1 separately generates images of two classical pokemons called pikachu and squirtle using WGAN.<br/>
Code_2 directly generates images of 14000 pokemons using DCGAN.<br/>
### 4.3 Usage
#### Code_1
<pre><code>python resize.py<br/>
python rgb.py<br/>
python GAN.py<br/></code></pre>
#### Code_2
You can use our dataset to train the model by<br/>
<pre><code>python main.py --input_height 64--input_width 64 --output_height 64 --output_width 64 --data pokemon --crop --train --epoch 80 --input_fname_pattern "*.jpg"</code></pre>
Also, you can use your own dataset by<br/>
<pre><code>$ mkdir data/DATA_NAME</code></pre>
Then, add you images into DATA_NAME folder by
<pre><code>python main.py --input_height **--input_width ** --output_height ** --output_width ** --data DATA_NAME --crop --train --epoch 80 --input_fname_pattern "*.jpg"</code></pre>
### 4.4 Result
Pikachu (295 pictures, 5000 epoches, 50 epoches per image)<br>
![](part2/results/Pikachu.gif)<br>
Squirtle (280 pictures, 5000 epoches, 50 epoches per image)<br>
![](part2/results/Squirtle.gif)<br>
Pokemons (14000 pictures, 80 epoches, 1 epoch per image)<br>
![](part2/results/Pokemon_all.gif)<br>
Final Result<br/>
![](part2/results/final_result.png)<br>
## 5. Credit
[Part_1 dataset](https://www.kaggle.com/terminus7/pokemon-challenge)<br/>
[Part_2 dataset](https://www.kaggle.com/thedagger/pokemon-generation-one)<br/>
[Part_1 code](https://github.com/llSourcell/Pokemon_GAN)<br/>
[Part_2 code](https://github.com/carpedm20/DCGAN-tensorflow)<br/>
[Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks](https://arxiv.org/abs/1511.06434)<br/>
https://medium.com/@yvanscher/using-gans-to-create-monsters-for-your-game-c1a3ece2f0a0<br/>
https://zhuanlan.zhihu.com/p/24767059<br/>
