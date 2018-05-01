# coursera-python-ml

## Week 1

### Key Concepts
- 2 key classes of ML algorithms
- **Supervised learning** (predict some output value that is associated with each input item). _Learning to predict target values from labelled data._ If the outout is a category with a finite and defined number of possibilities, then this is a classification task, and the function we train is a classifier. If the output variable is a number (e.g. amount of time in seconds, building energy usage), then this is a regression task, and we train a regression function.
- We typically denote a table of data items using X, with one data item per row. Labels we associate with each item are stored in variable y. _The goal is to learn some function that maps data item X into a label in Y_. To do this, the system is given a set of labelled training examples - inputs X _i_ and Y _i_, which are used to identify the function that best maps to the desired output.
- Where does the initial set of labelled data come from? Typically, from human judges.
- Can be easy or difficult, based on how much label data is needed, level of domain knowledge required, and complexity of the labelling task.
- Crowdsourcing platforms like Mechcanical Turk/CrowdFlower have been significant in connecting sponsors with groups of workers who can provide explicit labels using human intelligence.
- Implicit labels can also be inferred, for example search engine detects user clicking on a result and not returning for ~2 mins, that could be used as an implicit label for the returned page. i.e. longer on the page = more releveant to their query.
- **Unsupervised learning** for cases where we only have input data, and no labels to go with the data. The problems we solve here involve taking the input data and trying to find useful structure in it.
- Structure could be clusters or groups. e.g. segmenting users based on web traffic/clickstream data, then using this model/algorithm to tailor site offerings to each group to increase liklihood of purchase, provide better experience.
- _You don't know how many user groups you have_ ; typical unsupervised learning problem
- Unsupervised learning can also be used to train an algorithm to spot outliers (outlier detection). This doesn't assume future attacks will be the same form as previous attacks, but does assume that features of attacks will look different to how users generally interact. 
### Solving problems using ML
- Three basic steps
- **Figure out how to represent the problem in terms the computer can understand**, and choose the type of classifier that's appropriate. Get the data, formulate the description of your object that you're interested in recognising in a way that can be fed into the algorithm. e.g. an image could be reperesented as an array of coloured pixels, meta data .. metadata of a card transaction like time, place, amount of £ for the transaction.
- Also known as feature extraction/feature engineering. Attribute values are features of X. 
- **Decide on evaluation method** that provides a quality of accuracy score. Following training, we can do failure analysis to see where the algorithm is making mistakes, and use this to refine the feature set. 
- **Find the optimal model** that gives the best evaluation outcome for the problem. 

### Python tools for ML
- `scikit-learn, scipy, numpy, pandas, matplotlib` will be used here. `scikit-learn` is the most popular python library for ML, and forms the basis of this course. Many sample applications and code online.
- scikitlearn user guide and API reference are useful. 
- `import scipy as sp`
- We'll be using scipy to generate _sparse matrices_, big tables that consist mostly of 0s.
- numpy provides fundamental data structures (e.g. multi dimensional arrays). Typically data fed to scikit is in this format.

### An example problem
For reference: [Jupyter Notebook](https://dataplatform.ibm.com/analytics/notebooks/v2/18054cb0-8084-4bcd-9243-d0c9b3efb01d/view?access_token=3224314752b5beefeb1f4194e4b03eddd94bac83f759b1de2241324cf3f00c91)
- Object recognition system - simple but relects the same key concepts in real world systems.
- _Training a classifier to distinguish between different types of fruit_
- Recorded measurements in a table of different fruits. 
- Predicting fruit types might seem silly and moot .. but food companies do rely on ML techniques like this, for automated quality control, screening for bad fruit whilst processing etc. (Rotton fruit detection uses UV light that can detect interior decay, which is less visible on the surface .. but it's still taking an input and using this to score and make decisions)
- In supervised learning, the data will also include a label column - this can be derived from info in one or more columns.
- If we use data to train, we cannot use this same data to evaluate. Evaluation should always be done using unseen data.
- `train_test_split` randomly shuffles the data set and splits off a certain percentage for training, and puts the rest in a different variable for testing. Here we're doing 75/25 split, which is pretty standard.
- `X` for the different variables, `y` for the labels, e.g.:
```
X = fruits[['height', 'width', 'mass', 'color_score']]
y = fruits['fruit_label']
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
```
- `random_state` provides seed value. 
- Looking at the data set first (via visuals, or just having a scroll) is generally a good idea. 
    - Gives you a sense for what's in the dataset, and any other cleaning or pre-processing you might need to do
    - The range and distribution of values that is typical for each attribute
    - Particularly valuable when dealing with text that is represented by multiple features that have been extracted in pre-processing. 
    - Notice missing or noisy data
    - Data types
    - Not enough examples of a particular labeled class
    - Might not need to apply ML to this problem - there might be another way to classify in a machine readable format (e.g. geolocation stored in image metadata to infer location rather than training classifier)
- _Only use the test set for testing_ - this complete separation is important and will be covered later.
- Can use a `matplotlib cmap` (`from matplotlib import cm`) to plot scatters/histograms breaking down each feature by the labels.
- Well defined classes and separation in the feature space is a good indication that suggests the classifier is likely to be able to predict the class label from the features with good accuracy. Visualisation used in this module works well with feature sets of less than 20.
- Unsupervised learning has different vis techniques to cover large feature dimensions (hundreds, thousands, millions of features), to represent each object.
- Feature pair plot plots all possible pairs of features and produces scatter plot for each pair, showing whether features are correlated or not.
- Diagonal shows histogram with distribution of feature values for that feature.
- Can also look at features that use a subset of 3 features by building a 3d plot (this is awesome)

### K-nearest neighbors classification
- aka `k-NN`, can be used for classification and regression. An example of instance based or memory based supervised learning (memorize labeled examples from training set and use this to classify new objects later).
- `k` refers to the number of nearest neighbors the classifier will retrieve and use to make its predicition.
- k-NN has three steps that can be specified:
    - when it sees a previously unseen data object, it looks at the training set to find the k examples that have the closest features.
    - then looks up class labels for those k-NN examples
    - finally, combine the labels of those exampes to predict the label of the new object, typically, via majority vote.
- `query point` is the point you want to classify
- `decision boundaries` are the lines between one class region and the next. 
- decision boundaries are based on ecludian distance. 

## Week 2
### Intro to Supervised ML
- Looking beyond k-NN to other algorithms, how to apply them, and how to interpret their results
- _feature representation_ is taking an object (e.g. a piece of fruit) and converting it to numbers a machine can understand
- k-NN recap:
    - `from sklearn.model_selection import train_test_split` and `from sklean.neighbors import KNeighboursClassifier`
    - `X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)` where `X` is feature df and `y` contains corresponding labels
    - `knn = KNeighborsClassifier(n_neighbors = 5)`
    - `knn.fit(X_train, y_train)`
    - `knn.score(X_test, y_test)`
    - `knn.predict(example_feature_data)`
- Supervised learning can be divided into two types of tasks - classification and regression - both take a set of training data and learn mapping to a target value. Classification maps to a discrete class value (can be binary (yes or no), multi-class (fruit example), or multi-label (entitiy extraction)). Regression predicts a continuous variable.
- Relationship between model complexity and accuracy will be explored as we learn about the new algorithms. Generally, when measuring performance against the training set, more complex models will fit the training data better and better (obviously). However, when evaluating against a test set (generally good practice .. ), there is typically an initial accuracy gain from adding model complexity, but then a decrease in test set accuracy as the model becomes overfit to the training data, and doesn't capture more general trends / patterns which allow it to generalize to unseen data. 
- Statistically, input variables = independent variables and outcome variables = dependent variables.

### Overfitting and Underfitting
- successful supervised learning is gauged on the algorithms ability to predict on unseen data.
- algorithm will assume test set is drawn from same underlying distribution as training set - _overfitting_ typically occurs when we try to fit an overly complex model with insufficient data volumes. _The model can't see more general, global data patterns if the training set is too small_ - there is not enough data to constrain the model to respect the broader trends.
- Understanding, detecting, and avoiding overfitting is perhaps the most important aspect of applying supervised ML.
- Regression example: applying linear regression when the model needs to be more complex leads to underfitting. Quadratic relationship could give a curve, provide improved fit to the training data. If we believe the relationship between variables to be a function of several different parameters, a polynomial regression may fit better, and capture more subtle trends, but also has much higher varience, leading (potentially) to overfitting (too localized to the training data)
- For a simple 2 dimensional classification, the problem is in defining the decision boundary. A linear classifier can underfit (a la the above example), simply doesn't capture the patterns, and fit quite well, if it is aware of the location of certain data points. Overfitting would again, be overly complex and fit very well with the training data, but be too specific. Highly variable. 
- For k-NN, the general idea is that as we decrease K for k-NN classifiers, we increase the risk of overfitting because, where K=1 for example, we're trying to capture very local changes in the decision boundary that may not lead to good generalization behavior for future data. 

### Datasets
- `sklearn` has a variety of methods to create synthetic datasets in the `sklearn.datasets` lib.
- synthetic data sets typically have low number of features/dimensions. Makes them easy to explain and visualize. high dimensional (typically real world data sets are) datasets have most of their data in corners with lots of empty space, making it more challening to visualise
- `make_regression`, `make_classification`, `make_blobs`
- we'll use fruits data set for multi-class classification

### k-NN : Classification and Regression
- To make a classification predicition for any query point, the classifier looks back in its training set to identify the k neighbors.
- increasing k can result in a much smoother decision boundary, i.e. lower complexity and less variance - which can result in better performance on unseen test data, as global trends are better reflected in the model. 
- k-NN for regression works as you would expect - find the k neighbors, and the prediction = average y (continuous variable) of the k training points.
- an r^2 (r squared) value is used to assess how well the regression model fits the data.
- k-NN is clear and easy to understand why a prediticion was made. Can be a reasonable baseline with which to compare more complex models. When the training data has many instances/festures, or with sparse data (lots of features, but mostly 0's), k-NN can be slow. 
- _We have not explored the metric parameter_

### Linear Regression - Least-Squares
- Linear model expresses the target outputt in terms of a _sum of weighted inputs_.
- Each input feature is denoted x0, x1, etc, and each feature (xi) has a corresponding weight (wi). (This is similar/_the same?_ to evaluating inputs via weighted nodes in a single layer NN)
- Least-squares linear regression finds the line through the training data that minimizes the means squared error of the model - i.e. the sum of the squared differences between predicted targetand actual target for all points in the training set. It finds the slope (w) and y intercept (b) that minimizes the mean squared error.
- There are no parameters to control model complexity.
- Model complexity is based on the nature of the weights on the input features. Simple models have weight vector closer to zero (more features are not used at all, or have minimal impact on the outcome, i.e. a very small weight)
- Learning algorithm predicts the target value from each training example, then computs a loss function for each (penalty for incorrect predictions). Incorrect = predicted value is different than actual target value. An example - squared loss function would return the squared difference between target and actual as the _penalty_.
- Algorithm then computs/searches for the set of w,b params that minizies the total of the loss function against all training points.
- `from sklearn.linear_model import LinearRegression` to implement - e.g. `linreg = LinearRegression().fit(X_train, y_train)`
- then `linreg.coef_` = w weight, and `linreg.intercept_` = b. This is just y = mx + c.
- _If a scikitlearn object attribute ends with an underscore, it means these were derived from training data and not quantities set by the user_
- k-NN doesn't make a lot of assumptions about the data structure, so gives potentially accurate but sometimes unstable predictions that are sensitive to small changes in training data. Linear models make strong assumptions about data structure, providing stable but potentially inaccurate predictions. 

### Ridge, Lasso, Polynomial Regression
#### Ridge
- Ridge regression is another way to estimate w and b for a linear model. It uses the same least-squares criterion, but adds a penalty for large variations in w params. 
- Regularisation = the addition of the penalty term. This is useful because it prevents overfitting , so improves likely general performance of the model by restricting possible parameter selections. Regularisation usually reduces complexity of final estimated models. Reduced complexity = reduced weights, and regularisation supports this because bigger weights incur a higher penalty. Practically, this means that regression accuracy on problems with lots of features can be notably improved.
- The amount of regularisation to apply is controlled by the alpha parameter (L2 penalty). Larger = more regularisation. (default = 1.0). Setting alpha to 0 is what we were working with earlier, minimizing to total squared error.
- `from sklearn.linear_model import Ridge`, then use the estimator object as we did before - e.g. `linridge = Ridge(alpha=20.0).fit(X_train, y_train)`
- Regularisation becomes less important as volume of training data increases.
- We can get better results from ridge regression by applying _feature preprocessing and normalization_.
- Because regularisation is imposing the sum of squares penalty on the size of the weights, and the scales of the different features can be very different, then different scales have different impacts on the total penalty incurred (because it's a sum of squares) - transforming the input features to be on the same scale means the ridge penalty is more evenly applied and generates better results.
- Bottom line? _Normalisation is important_, and will study more as we go. 
- *MinMax scaling* is a widely used form of feature normalisation. This transforms all input variables so they are on a scale between 0 and 1. This is done by taking the min and max values from each feature on the training data, then applying the minmax formula:
!(formula)[img/minmax.png]
- Example use:
```
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)
clf = Ridge().fit(X_train_scaled, y_train)
r2_score = clf.score(X_test_scaled, y_test)

# Can do fitting and transformation together on training set using
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
```
- Things of nooooote:
    - we apply the same scalar object to both training and test set
    - we train the scaler on the training and not the test data
    - *these are critical!* If the same scaling is not applied to training and test sets, you'll have random data skew and invalid results. If the scalar is prepared (or other normalisation method) with the test set, we get Data Leakage (training phase has information that has leaked from the test set).
- Downside is that transformed features can be harder to interpret.
#### Lasso
- Lasso also adds regularisation penalty, this one is L1. Looks similar, but is a sum of the absolute values, rather than sum of squares. 
- This sets parameter weights to 0 for least influential variables (called sparse solution), is a kind of feature selection.
- alpha controls L1 regularization (still not quite got my head around alpha..) - there is an optimal range for alpha that neither under or over fits (of course different for each data set), and other factors like preprocessing methods
- Use *ridge* when you have many small/medium sized effects on output variables. Use *lasso* when you have only a few variables with medium/large effect on output.
- `Lasso` much the same to implement as `Ridge` above, but has a `max_iter` arg, this will increase computation time, but will assist with convergence warnings. 
- can look at features with non-zero weight to understand heavy weights and strong relationships between input variables.
#### Polynomials
- Taking two data points, multiplying them all by each other in every way to get 5 dims, and now write a new regression problem to predict y^ but with these 5 features instead of 2.
- _This is still a a linear regression problem_, the features are just numbers within a weighted sum.
!(formula)[img/polynomial.png]
- ^ Polynomial feature transformation, transforms a problems into a higher dimensional regression space, and allows us to use a richer set of complex functions to fit the data. _This is very effective with classification_. `from sklearn.preprocessing import PolynomialFeatures`
- When we add these features, we're adding to the models ability to capture interactions between the different variables by adding them as features to the model.

### Logistic Regression
 - (Actually used for classification!)
- :FUTURE READING: - Non-linear basis functions for regression.
