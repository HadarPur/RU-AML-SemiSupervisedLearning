# Semi-Supervised Learning 

Timot Baruch, Hadar Pur

Submitted as a project report for Advanced Machine Learning course, IDC, 2024

## Dataset Generation

### Conditionally Independent Dataset - Bonus

To describe the data generation process for the semi-supervised learning task, we utilize a synthetic dataset
creation method. The process involves generating two views of the data, each containing information relevant to the classification task. Here’s a detailed description of the process.

#### Initialization

In the initialization phase, we set the parameters for our dataset, specifying the number of samples as 1000 and the number of features as 10. Following this, we construct covariance matrices to capture the variability and relationships among features. The first covariance matrix, named covariance1, serves to denote no correlation between features, being an identity matrix. Conversely, covariance2, also an identity matrix, scales its diagonal elements by a factor of 2, indicating higher variability.
Next, we establish mean vectors, mean1 and mean2, representing the central or average values for each feature. Mean1 is initialized with zeros, suggesting low average values across all features. In contrast, mean2 is set to 0.7 for all features, reflecting a shift towards higher values compared to mean1.

#### Data Generation

The dataset is then split into two halves (5 features each) to accommodate binary classification, allocating an equal number of samples to each class. Data generation involves creating samples for each class (0 and 1) in both views using multivariate normal distributions. For class 0, samples are generated with mean1 and covariance1, signifying low variability centered around zero. Conversely, for class 1, samples are generated with mean2 and covariance2, representing higher variability centered around 0.7.
Following data generation, samples for each class in both views are concatenated, forming the complete datasets, view1 and view2. Subsequently, labels are assigned based on class membership, with the first half of samples in the concatenated datasets assigned to class 0 and the second half to class 1.
Finally, the dataset undergoes shuffling to randomize the order of samples within each class, ensuring uniform distribution and eliminating potential biases in subsequent analyses or model training. This data generation
process results in two views of the dataset, each containing samples with distinct features and class labels, while keeping the two sets of variables independent given the class label.

<p align="center">
<img src="https://github.com/HadarPur/RU-AML-SemiSupervisedLearning/blob/main/figs/fig1.png" alt="drawing" width="800"/>
</p>

#### Conditional Independence Across Views in Dataset Generation

Conditional independence across views given the class label is ensured by:
1. **Class-Conditional Distribution**: For each class, samples are drawn from its own multivariate nor- mal distribution, with unique means and covariances defining each distribution. This approach cap- tures both the common characteristics within each class and the distinguishing features between classes. Consequently, the feature distribution across views within each class is entirely determined by these class-specific parameters.
2. **Independent Views**: A crucial element of this data generation method is the autonomy of views for each class. Features in one view are generated without being affected by or related to those in another view. This ensures that understanding the distribution of one view based on the class label does not provide any additional information about the other view, confirming their independence.

This engineered conditional independence is highly beneficial for multi-view learning algorithms. It enables these algorithms to utilize information from different views separately, improving their accuracy and relia- bility in classification tasks. Please note that in the notebook we made two statistic exams for determine weather or not the dataset has conditionally independent two sets of variables (the two views) given the class labels, as required.

### Provided Dataset

We also utilized the dataset provided by the class instructor, which has similar characteristics as the previous dataset. It comprises 1000 samples labeled with binary classes 0 or 1, and each view contains 5 features. Notably, the features are presumed to be conditionally independent given the class label.

<p align="center">
<img src="https://github.com/HadarPur/RU-AML-SemiSupervisedLearning/blob/main/figs/fig2.png" alt="drawing" width="800"/>
</p>

### Data Pre-Processing

In this step, we divided the dataset into training, validation, and test sets where 70% of the dataset is used for training, 15% is for testing and 15% is for validation. Then, we removed 90% of the labels from the training set by randomly selecting a subset of labels and marking them as unlabeled (-1). This process aimed to simulate a semi-supervised learning scenario where a large portion of the training data lacks class labels.

## Co-training Implementation

### Our Co-training Algorithm

Co-Training is a semi-supervised learning technique that leverages the information contained in multiple views of the data to improve classification performance, particularly in scenarios where labeled data is scarce.
We implemented the Co-Training algorithm inspired by the article by Blum & Mitchell [1]. The implemen- tation follows the pseudo code outlined below:

<img src="https://github.com/HadarPur/RU-AML-SemiSupervisedLearning/blob/main/figs/alg.png" alt="drawing" width="800"/>

The Co-Training process involves several iterative steps, which are outlined below:
1. **Initialization**: Two base classifiers, denoted as model1 and model2, are initialized using labeled data available initially.
2. **Iteration**:

  * **Training**: We trained two base classifiers, using the labeled data available initially. This step involved fitting the classifiers to the labeled data, allowing them to learn the underlying patterns and relationships between the features and labels.
  * **Pool of Unlabeled Data**: Initially, a pool of unlabeled data, was created by randomly selecting a subset of pool size unlabeled examples from the dataset (similar to U′ in the article). After pseudo-labeling in each iteration, this pool was reoccupied by randomly choosing samples from the remaining unlabeled dataset.
  * **Prediction**: After training the classifiers, we generated predictions for the unlabeled data points in the pool of unlabeled samples, using both model1 and model2. This step involved using the trained classifiers to predict the class labels for the unlabeled samples, even though their true labels were unknown.
  * **Pseudo-Labeling**: In this step, we pseudo-labeled the unlabeled data points based on the clas- sifiers’ predictions. We utilized a parameter called top k (which serves the same purpose as the p and n parameters in the article) to determine how many of the highest probability samples were chosen for labeling in each iteration of the co-training loop. Samples with high confidence (exceeding a predefined threshold denoted by confidence level parameter) were selected to be added to the labeled dataset.
  * **Update**: After pseudo-labeling the unlabeled data points, we augmented the labeled dataset with the newly pseudo-labeled samples. These samples were then removed from the pool of unlabeled data to prevent duplication and ensure the integrity of the training process.
  * **Retraining**: Finally, we retrained both classifiers using the updated labeled dataset, which now included the newly pseudo-labeled samples. This step involved fine-tuning the classifiers based on the augmented labeled data to further improve their performance and adapt to the newly acquired information.

3. **Our Model’s Parameters**: Our model’s parameters consist of several key components. Firstly, we have model1, which represents the first classifier utilized within the co-training algorithm, and model2, which denotes the second classifier employed in the same algorithm. Additionally, there are optional parameters such as num iterations, specifying the number of iterations with a default value of 100, and pool size, indicating the size of the pool of unlabeled samples with a default of 70. Another parameter is the confidence level, determining the confidence level for label assignment, set to a default of 0.65. Lastly, the top k parameter dictates the number of highest probability samples chosen for labeling in each iteration of the co-training loop, with a default value of 4. Throughout our experiments with both datasets, we choose these default parameters because they achieved the best classification results, while trying different models, as detailed in subsequent sections.
4. **Termination Condition**: The iterative process continues until reaching max iterations or until all the unlabeled data is being labeled. We allow for maximum of pool size labels to be unlabeled (due to not exceeding the threshold number). We wanted to classify only the most certain samples. Reaching a confidence level below 0.65 (default value for confidence level) implies that the model is uncertain about how to classify a particular sample and it’s not a definitive classification.
5. **Evaluation**: We evaluate each co-training iteration on the validation dataset.
The Co-Training process capitalizes on the diversity of information provided by multiple views of the data and iteratively refines the classifiers’ predictions by incorporating pseudo-labeled data. This approach effectively utilizes both labeled and unlabeled data to enhance classification accuracy, making it particularly useful in scenarios where obtaining labeled data is challenging or expensive.

### Training on Generated Dataset

In order to train our co-training classifier on the generated dataset, we chose the Naive Bayes classifier (for both models) after experimenting with various classifiers because of its suitability for conditionally indepen- dent datasets. We used default parameters as provided in section 3. The results are presented below.

<p align="center">
<img src="https://github.com/HadarPur/RU-AML-SemiSupervisedLearning/blob/main/figs/table1.png" alt="drawing" width="800"/>
</p>

<p align="center">
<img src="https://github.com/HadarPur/RU-AML-SemiSupervisedLearning/blob/main/figs/fig3.png" alt="drawing" width="800"/>
</p>

### Training on Provided Dataset

In order to train our co-training classifier on the provided dataset, we chose the Random Forest classifier as model1 and the SVM classifier as model2 after experimenting with various classifiers and achieving best results with this combination. We wanted to combine two different classifiers to observe how our co-training algorithm handles two distinct models, aiming to create a more robust model while hoping to achieve better classification results. We used default parameters as provided in section 3. The results are presented below.

<p align="center">
<img src="https://github.com/HadarPur/RU-AML-SemiSupervisedLearning/blob/main/figs/table2.png" alt="drawing" width="800"/>
</p>

<p align="center">
<img src="https://github.com/HadarPur/RU-AML-SemiSupervisedLearning/blob/main/figs/fig4.png" alt="drawing" width="800"/>
</p>

## Evaluation & Comparisons

### Generated Dataset

We assessed the performance of our co-training approach by removing 90% of the labels. We utilized Naive Bayes classifiers as our models. This was compared against the fully labeled dataset trained with the Naive Bayes classifier. Below are the results of this comparison.

<p align="center">
<img src="https://github.com/HadarPur/RU-AML-SemiSupervisedLearning/blob/main/figs/fig5.png" alt="drawing" width="800"/>
</p>

We observed that our model, even when trained with only 10% of the labels, achieved competitive results compared to models trained on fully labeled data. Specifically, our model attained an accuracy of 85% on the test data, whereas the fully labeled model achieved 89% accuracy.
We also examined the decision boundaries of our co-training classifier on the test and train sets. We observe that the algorithm handles the data well despite the absence of 90% of the labels. It can still distinguish effectively between the different classes, which is impressive given that the features are conditionally inde- pendent given the labels.

<p align="center">
<img src="https://github.com/HadarPur/RU-AML-SemiSupervisedLearning/blob/main/figs/fig6.png" alt="drawing" width="800"/>
</p>

### Provided Dataset

We assessed the effectiveness of our co-training approach by removing 90% of the labels. We utilized Random Forest as our first model and SVM classifier as our second model. We compared this against models trained with the fully labeled dataset. The results of this comparison are outlined below.

<p align="center">
<img src="https://github.com/HadarPur/RU-AML-SemiSupervisedLearning/blob/main/figs/fig7.png" alt="drawing" width="800"/>
</p>

We observed that our model, even when trained with only 10% of the labels, achieved competitive results compared to models trained on fully labeled data. Specifically, our model attained an accuracy of 93% on the test data, whereas the fully labeled SVM achieved 92% accuracy and the fully labeled RF achieved 95% accuracy.

In the provided dataset, similar to the generated dataset, we examined the decision boundaries created by our co-training algorithm. We observed well-defined contours of the different classes, indicating the effec- tiveness of the algorithm in classification. These results were particularly surprising considering that our co-training classifier combines two different models to create the decision boundaries.

<p align="center">
<img src="https://github.com/HadarPur/RU-AML-SemiSupervisedLearning/blob/main/figs/fig8.png" alt="drawing" width="800"/>
</p>

## Visualization - Bonus

In this section, we created an animation to visualize how our Co-Training classifier learns from the data over time. We used it to visualize the process on the generated dataset. Here’s a simplified explanation:
1. Update Function: We defined a function called update that updates the animation frame by frame. This function adjusts the decision boundaries of our classifier and plots them on two separate views of the data.
2. Animation Creation: We used Matplotlib’s FuncAnimation class to create the animation. It iterates through different frames, with each frame representing a step in the Co-Training process. The update function is called for each frame to update the plot accordingly.

<p align="center">
<img src="https://github.com/HadarPur/RU-AML-SemiSupervisedLearning/blob/main/images/ani.gif" alt="drawing" width="800"/>
</p>

## Summary

In this project, we explored the application of Co-Training, a semi-supervised learning technique, in the context of binary classification tasks. We began by generating a synthetic dataset with conditionally inde- pendent features, simulating a scenario where labeled data is limited. The dataset consisted of two views, each containing distinct feature sets, and binary class labels of 0/1. Through careful data generation, we ensured that the features across views remained conditionally independent given the class label. We also used the provided dataset’ which is also a binary dataset 0/1 consists of conditionally independent variables given the labels. we removed 90% of the labels for the co-training process.
We then implemented the Co-Training algorithm, following the framework outlined in the work by Blum & Mitchell [1]. Our implementation involved iteratively training two base classifiers, pseudo-labeling unlabeled data points, and updating the classifiers using the augmented labeled dataset. We introduced parameters such as the number of iterations, pool size, confidence level, and top-k for fine-tuning the algorithm’s behavior.
Our experiments involved training the Co-Training algorithm on both the generated dataset and a provided dataset. For the generated dataset, we employed Naive Bayes classifiers as our models and observed com- petitive performance even with 90% of the labels removed. On the provided dataset, we utilized Random Forest and SVM classifiers, achieving comparable results to models trained on fully labeled data.
Notably, our Co-Training approach demonstrated robustness and effectiveness in handling datasets with lim- ited labeled data. Even when compared to models trained on fully labeled datasets, our approach achieved very similar accuracy, precision, and recall results on the validation data. By leveraging two different views of the data and iteratively refining classifiers, we were able to achieve competitive classification accuracy, highlighting the potential of semi-supervised learning techniques in real-world scenarios where labeled data is scarce or expensive to obtain. Additionally, our visualization of decision boundaries showcased the algo- rithm’s ability to create well-defined contours of different classes, further validating its efficacy in classification tasks.

## Bibliography
[1] Avrim Blum and Tom Mitchell. “Combining Labeled and Unlabeled Data with Co-Training”. In: Pro- ceedings of the Annual ACM Conference on Computational Learning Theory (Oct. 2000). doi: 10.1145/ 279943.279962.
