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
- 
