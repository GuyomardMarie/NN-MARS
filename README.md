# NN-MARS

This repository contains the algorithm of the NN-MARS.

**Please cite our associated papers:**

Article [1] (English) : Adaptive splines-based logistic regression with a ReLU neural network, Marie Guyomard, Susana Barbosa, Lionel Fillatre, JOBIM, 2022. ⟨hal-03778323⟩ 

Article [2] (French) : Régression logistique à base de splines adaptatives avec un réseau de neurones ReLU, Marie Guyomard, Susana Barbosa, Lionel Fillatre, GRETSI, 2022. ⟨hal-03778328⟩

**We provide here:**

The different classes and functions in order to compute NN-MARS in two forms : 'NNMARS_Functions.py' and 'NNMARS_Functions.ipynb'
An example of ho to use the NN-MARS : 'NNMARS_main.html' (compiled) and 'NNMARS_main.ipynb'


**About the NN-MARS:**

The proposed algorithm is available here for **binary** classification problems. 

Let suppose we have $N$ independant and identically distributed pairs ${x^{(i)}, y^{(i)}}_{i=1}^{N}$. We consider the following neural network :

$$f(x) = \hat{\mathbb{P}}\left(Y=1|X\right) = \sigma(\psi_X(\theta)).$$
	

with 

$$\sigma = \frac{1}{1+\exp(-X)} $$
and 
$$\psi_X(\theta) = \beta_0 + \beta \phi\left(\bar{W}X^T+b\right).$$

such as

$$\phi(X) = \text{ReLU}(X) = \max\{0,X\}$$

$$X \in\mathbb{R}^{N \times d} $$

$$b = [b_{11}, b_{12}, \dots, b_{d1}, b_{d2}]^T \in \mathbb{R}^{2d} $$

$$\beta = [\beta_{11}, \beta_{12}, \dots, \beta_{d1}, \beta_{d2}] \in \mathbb{R}^{2d} $$

$$\bar{W} = \begin{bmatrix}
		-1 & 0 & \dots & 0 \\
		1 & 0 & \dots & 0 \\
		0 & -1 & \dots & 0 \\
		0 & 1 & \dots & 0 \\
	\end{bmatrix} \in \mathbb{R}^{2d \times d} \\$$
	
$$\theta = [\beta_0, \beta, b^T]^T = [\beta_0, \beta_1, \dots, \beta_d, b_1, \dots, b_p]^T $$


<img width="1217" alt="Schema_NNMARS" src="https://user-images.githubusercontent.com/93378786/191957351-97269ef7-e9fe-4520-ab8b-8bceb7fe64d0.png">

*Remark : for binar categorical variable, only one node is created.*
