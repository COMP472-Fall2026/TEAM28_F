# Technical Report COMP472 Team 28

Karim Mikhaeil 40233685

Ashwin Vazhayil 40346223

Victor Depaz 40242703

# Model Architecture and Training

## k-Nearest Neighbours

### Architecture and Implementation

We have implemented k-NN using NumPy, the classifier stores the training feature vectors and their labels then it computes the distances to each query, and it predicts its class using the nearest training examples. The implementation supports euclidean and cosine distances with either uniform voting or inverse distance weighted voting.

### Training Methodology

Oppositely to a neural network, k-NN does not learn weights through iterative optimization, instead its fitting step stores the reference examples, therefore epochs, learning rate, and a training loss function do not apply.

| Hyperparameter | Tested values |
|---|---|
| Number of neighbours | 1, 3, 5, 10, 20, 50, 100, 200 |
| Distance metric | euclidean, cosine |
| Voting | Uniform, inverse-distance-weighted |
| Preprocessing | Raw features, standardized features |

Standardization parameters are fitted on training data only. Configurations are compared using validation accuracy.

Additional image experiments apply an element-wise square root followed by L2 normalization.

### Local-Mean Variant

We also implemented a local-mean variant that computes a score for each class by averaging its highest training-example similarities. The image experiments compare 2, 3, and 5 neighbours per class, while the sentiment experiments compare 25 and 50.

The final image pipeline uses 2 neighbours per class. Class scores are converted into probabilities using softmax with a temperature of 0.02 before combining them with MLP predictions.


## Decision Tree

### Architecture and Implementation

We implemented a binary Decision Tree from scratch using NumPy. At each node, the algorithm examines candidate feature thresholds and selects the split with the lowest weighted child impurity.

The implementation supports Gini impurity and entropy. Each leaf stores a predicted class and the class distribution of its training examples, allowing both label and probability predictions.

### Training Methodology

Training recursively partitions the data until a stopping condition is reached. These conditions include maximum depth, insufficient samples, a pure node, or the absence of an impurity-reducing split.

The experiment script evaluates the following configurations:

| Maximum depth | Minimum samples to split | Minimum samples per leaf | Criterion |
|---:|---:|---:|---|
| 5 | 10 | 5 | Gini |
| 10 | 10 | 5 | Gini |
| 15 | 10 | 5 | Gini |
| 20 | 10 | 5 | Gini |
| 25 | 10 | 5 | Gini |
| 20 | 20 | 5 | Gini |
| 20 | 10 | 2 | Gini |
| 20 | 10 | 5 | Entropy |

Standalone trees consider all features when searching for a split. Training uses greedy impurity reduction rather than gradient descent, so epochs and learning rate do not apply.


## Random Forest

### Architecture and Implementation

Our Random Forest is implemented from scratch using our Decision Tree classifier. Each tree is trained on a bootstrap sample drawn with replacement from the training data.

At each node, a randomly selected feature subset is considered for splitting. Predictions are combined using either hard voting, where each tree contributes a class vote, or soft voting, where leaf class probabilities are averaged.

### Training Methodology

The raw-feature experiments use the following configurations:

| Hyperparameter | Images | Sentiments |
|---|---:|---:|
| Number of trees | 25 | 50 |
| Maximum depth | 20 | 15 |
| Minimum samples to split | 10 | 10 |
| Minimum samples per leaf | 5 | 5 |
| Features considered per node | 22 | 27 |
| Splitting criterion | Gini | Gini |
| Random seed | 42 | 42 |

The number of candidate features is the integer part of the square root of the input dimension. Intermediate forest sizes are evaluated after 10, 25, and, where applicable, 50 trees.

Hard and soft voting are compared using exactly the same trained trees, isolating the effect of prediction aggregation.

Additional experiments use regularized Fisher projections fitted on training data only. These forests contain 50 trees; the configurations include depth 18 with a minimum leaf size of 5, and a sentiment variant with depth 5 and a minimum leaf size of 10.

Random Forest training does not use epochs or a learning rate. Each constituent tree is fitted using greedy Gini impurity reduction.


## Multi-Layer Perceprton

### Architecture and Implementation

We implemented feed-forward MLPs using PyTorch. Each hidden layer consists of a fully connected transformation followed by a ReLU activation, with optional dropout.

The output layer contains 50 units for image classification and 3 units for sentiment classification. It returns logits, which are converted into probabilities using softmax during prediction.

The main architectures are:

| Configuration | Architecture | Hidden dropout | Input dropout |
|---|---|---:|---:|
| Baseline image MLP | 500 → 256 → 64 → 50 | 0 | 0 |
| Regularized image MLP | 500 → 1024 → 50 | 0.5 | 0.2 |
| Sentiment MLP | 768 → 64 → 3 | 0.3 | 0 |

Input dropout randomly masks input features during training. Hidden dropout randomly suppresses hidden activations to reduce reliance on individual units.

### Training Methodology

Networks are trained using mini-batch gradient descent with the Adam optimizer and cross-entropy loss.

| Hyperparameter | Image baseline | Regularized image MLP | Sentiment MLP |
|---|---:|---:|---:|
| Optimizer | Adam | Adam | Adam |
| Learning rate | 0.001 | 0.001 | 0.0003 |
| Weight decay | 0 | 0.001 | 0.01 |
| Batch size | 128 | 128 | 128 |
| Maximum selection epochs | 50 | 50 | 50 |
| Early-stopping patience | 8 | 8 | 8 |
| Loss function | Cross-entropy | Cross-entropy | Cross-entropy |

Features are standardized using statistics calculated on the training partition. The same transformation is then applied to validation examples.

During epoch selection, training stops after eight consecutive epochs without an improvement in validation accuracy. The model restores the weights from the epoch with the highest validation accuracy, rather than using the final epoch automatically.

Image experiments train each architecture with seeds 42, 123, and 2026. Ensemble predictions are obtained by averaging the networks' class probabilities.


## Local-mean k-NN

### Architecture and Implementation

We implemented k-Means from scratch using NumPy. The algorithm alternates between assigning examples to their nearest centroid and updating each centroid to the mean of its assigned examples.

The implementation uses k-means++ initialization and multiple independent restarts. The retained solution is the one with the lowest inertia, defined as the sum of squared distances between examples and their assigned centroids.

### Training Methodology

The clustering experiment script uses the following default settings:

| Hyperparameter | Setting |
|---|---|
| Initialization | k-means++ |
| Number of restarts | 3 |
| Maximum iterations per restart | 200 |
| Centroid movement tolerance | 0.0001 |
| Random seed | 42 |
| Initial cluster counts | 50 for images; 3 for sentiments |

Additional experiments vary the number of clusters. Training stops when assignments no longer change, centroid movement falls below the tolerance, or the iteration limit is reached.

Clustering is fitted on the provided labelled training examples without using their labels to update centroids. Labels are subsequently used to evaluate clustering quality; the script also uses NMI to select the representation for further cluster-count experiments.

Unlike the supervised classifiers, k-Means minimizes inertia rather than classification loss and does not use a learning rate.


# Evaluation

### 1. Evaluation Protocol and Metrics

The main classification experiments use a stratified 80/20 training–validation split. This provides 16,000 training examples and 4,000 validation examples for images, and 9,600 training examples and 2,400 validation examples for sentiments.

Preprocessing parameters are fitted on the training partition only. Validation data is used to compare configurations and select hyperparameters, while hidden test labels are unavailable.

Our primary evaluation metric is classification accuracy:

Accuracy = Number of correct predictions / Total number of predictions


We selected accuracy because it is simply the competition metric and both labelled datasets have balanced classes.

The competition gives equal importance to the two tasks:

Overall score = ( Image accuracy + Sentiment accuracy ) / 2

We also use precision, recall, macro F1-score, and confusion matrices to examine class-level performance. Macro F1 gives each class equal weight, while recall helps identify classes for which many examples are missed.


### 2. Comparative Performance

The following table summarizes the holdout validation accuracies reported in our project README.

| Model or configuration | Image accuracy | Sentiment accuracy |
|---|---:|---:|
| k-NN with cosine distance | 53.4% (57.2% with square-root features) | 58.4% |
| Decision Tree | 11.3% | 48.3% |
| Random Forest with raw features | 17.0% (26.2% on Fisher features) | 59.8% (67.1% on Fisher features) |
| MLP | 45.2% (44.2% on square-root features) | 68.3% |
| Local-mean k-NN | 59.1% | 58.8% |


#### Image Classification

The strongest individual image method in this comparison was local-mean k-NN, with 59.1% validation accuracy.

Standard cosine k-NN achieved 53.4%. Applying square-root features increased its accuracy to 57.2%, an improvement of 3.8 percentage points. This suggests that reducing the dominance of large histogram values makes neighbourhood comparisons more useful for these image features.

The Decision Tree and raw-feature Random Forest achieved lower accuracies of 11.3% and 17.0%, respectively. A possible explanation is that individual axis-aligned splits capture the useful relationships in these feature vectors less effectively than similarity-based methods.

Fisher features increased Random Forest accuracy to 26.2%. This demonstrates the importance of the representation, although this configuration remained below the strongest k-NN and MLP results.

#### Sentiment Classification

The MLP achieved the highest reported sentiment accuracy at 68.3%. Its hidden layer can learn combinations of the provided embedding dimensions that help distinguish the three sentiment classes.

The Random Forest improved from 59.8% on raw features to 67.1% on Fisher features, a gain of 7.3 percentage points. The resulting representation therefore made the sentiment classes easier to separate with the tested forest.

The Decision Tree achieved 48.3%, compared with 59.8% for the raw-feature Random Forest. This is consistent with the benefit of combining multiple randomized trees, although the configurations also differ in their training settings.

Local-mean k-NN achieved 58.8%, compared with 58.4% for standard cosine k-NN. Its improvement was much smaller for sentiments than for images.

### 3. Effects of Tree Depth

Maximum depth controls the complexity of a Decision Tree. A shallow tree may stop before capturing useful distinctions, while a deep tree can create highly specific rules that fit training examples but generalize poorly.

Our Decision Tree experiments compare depths of 5, 10, 15, 20, and 25 while keeping the minimum split size, minimum leaf size, and Gini criterion fixed for this depth comparison.

Training and validation accuracies must be considered together. An increase in training accuracy without a corresponding validation improvement indicates that the additional complexity is not improving generalization.

| Maximum depth | Image training accuracy | Image validation accuracy | Sentiment training accuracy | Sentiment validation accuracy |
|---:|---:|---:|---:|---:|
| 5 | 5.63% | 5.53% | 53.66% | 48.25% |
| 10 | 8.96% | 6.80% | 77.99% | 47.58% |
| 15 | 13.19% | 9.38% | 87.57% | 46.50% |
| 20 | 17.41% | 10.25% | 88.41% | 46.13% |
| 25 | 21.62% | 11.25% | 88.41% | 46.13% |

All configurations use raw features, Gini impurity, a minimum of 10 samples to split a node, and a minimum of 5 samples per leaf. Only maximum depth changes.

For image classification, increasing depth improves both training and validation accuracy. The best tested depth is 25, reaching 11.25% validation accuracy compared with 5.53% at depth 5. However, the training–validation gap also increases, indicating that some additional complexity fits training-specific patterns. Since validation accuracy continues to improve, these results do not establish that depth 25 is the optimal depth beyond the tested range.

For sentiment classification, the best tested depth is 5, with 48.25% validation accuracy. Increasing depth to 20 raises training accuracy from 53.66% to 88.41%, while validation accuracy decreases to 46.13%. This is clear evidence of overfitting: deeper trees fit the training examples more closely but generalize less effectively.

Depths 20 and 25 produce identical reported accuracies for sentiments, showing no additional benefit from increasing the depth limit. Overall, the effect of depth depends on the task: greater depth remains useful within the tested image configurations, whereas a shallower tree generalizes better for sentiments.

### 4. Effects of MLP Depth, Layer Size, and Regularization

We compared two image MLP architectures using the same validation split and standardized features. Each configuration was trained with three random seeds, and its ensemble averages the three networks' probabilities.

| Configuration | Hidden layers | Regularization | Ensemble validation accuracy |
|---|---|---|---:|
| Baseline | 256 → 64 | None | 37.625% |
| Wider regularized network | 1,024 | Input dropout 0.2, hidden dropout 0.5, weight decay 0.001 | 45.050% |
| Combined ensemble | All six networks | As above | 43.075% |

The wider regularized ensemble improved validation accuracy by **7.425 percentage points** over the baseline. Its individual networks also performed better across all three seeds, reaching 41.875–42.750%, compared with 33.425–33.775% for the baseline.

Both configurations showed overfitting. The baseline networks eventually reached 100% training accuracy while validation accuracy remained around 33–34%. The available histories for the wider networks also showed training accuracy near 100%, but higher validation accuracy, indicating better generalization without eliminating overfitting.

Combining all six networks achieved 43.075%, below the wider ensemble alone. Therefore, adding more networks did not automatically improve performance.

Overall, the wider regularized configuration performed better than the deeper baseline. However, width, depth, and regularization changed together, so the improvement cannot be attributed to one factor alone. Early stopping retained the checkpoint with the highest validation accuracy.
### 5. Effects of Preprocessing

Preprocessing had a substantial influence on performance, but its effect depended on the classifier.

For image k-NN, square-root features improved accuracy from 53.4% to 57.2%. In contrast, the reported image MLP accuracy decreased from 45.2% to 44.2% with square-root features.

This comparison shows that a transformation that improves distance-based neighbourhoods does not necessarily improve a neural network by the same amount.

Fisher projections improved the reported Random Forest results for both tasks:

| Task | Raw features | Fisher features | Difference |
|---|---:|---:|---:|
| Images | 17.0% | 26.2% | +9.2 percentage points |
| Sentiments | 59.8% | 67.1% | +7.3 percentage points |

These projections use training labels to construct a more discriminative representation. They were fitted without using validation labels.

### 6. Summary of Findings

The experiments show that model selection must consider both the classification algorithm and the feature representation.

For images, similarity-based methods performed particularly well, and the final local-mean k-NN–MLP blend achieved the highest reported validation accuracy of 62.6%.

For sentiments, the MLP achieved the highest reported accuracy at 68.3%, while Fisher projections substantially improved Random Forest performance.

The main findings are that preprocessing can strongly affect results, greater model complexity does not automatically improve generalization, and combining models with complementary predictions can outperform the strongest individual component.
