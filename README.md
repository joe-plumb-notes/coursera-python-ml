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
- `scikit-learn, scipy, numpy, pandas, matplotlib` will be used here. `skikit-learn` is the most popular python library for ML, and forms the basis of this course. Many sample applications and code online.
- scikitlearn user guide and API reference are useful. 
- `import scipy as sp`
- We'll be using scipy to generate _sparse matrices_, big tables that consist mostly of 0s.
- numpy provides fundamental data structures (e.g. multi dimensional arrays). Typically data fed to scikit is in this format.
