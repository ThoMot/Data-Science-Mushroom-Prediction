
# What is the danger of shrooming? 

- **Thora Mothes**
- **Mathias Ahrn**


```python
# code in this cell from: 
# https://stackoverflow.com/questions/27934885/how-to-hide-code-from-cells-in-ipython-notebook-visualized-with-nbviewer
from IPython.display import HTML
HTML('''<script>
code_show=true; 
function code_toggle() {
 if (code_show){
 $('div.input').hide();
 } else {
 $('div.input').show();
 }
 code_show = !code_show
} 
$( document ).ready(code_toggle);
</script>
<form action="javascript:code_toggle()"><input type="submit" value="Click here to display/hide the code."></form>''')
```




<script>
code_show=true; 
function code_toggle() {
 if (code_show){
 $('div.input').hide();
 } else {
 $('div.input').show();
 }
 code_show = !code_show
} 
$( document ).ready(code_toggle);
</script>
<form action="javascript:code_toggle()"><input type="submit" value="Click here to display/hide the code."></form>



## Introduction:

Shrooming refers to the act of looking for mushrooms or mushroom hunting. In this report predictions will be made as to how likely it is for a mushroom with certain characteristics to be poisonous or edible.

##### Dataset:
The chosen dataset for this report is called **<a href="https://archive.ics.uci.edu/ml/datasets/Mushroom">Mushrooms</a>**, and is a dataset consisting of descriptions of hypothetical samples corresponding to 23 species of mushroom in the Agaricus amd Lepiota Family. The data originally stems from the Audubon Society Field Guide to North American Mushrooms published in 1981, and was donated to <a href="https://archive.ics.uci.edu/ml/index.php">The UCI repository</a> in 1987. 

The dataset used in this report was downloaded from <a href="https://www.kaggle.com/uciml/mushroom-classification">Kaggle</a> as the UCI repository could not provide a CSV file. However the data from Kaggle has not been modified in any other way than formatting it into a CSV file. 

- Date downloaded: **12.09.2019**
- Hosted: **Thora Mothes GitHub Repo**
 

##### Goal:
In this report the goal is to predict whether or not a mushroom is edible. The feature used for output values is called "Class" and consists of values **e** and **p**, e for edible, and p for poisonous.

##### Features: 

When it comes to the features we will use for our models we will have two main criteria, we will try to find features that are easy for people to identify when they are looking for mushrooms and still give good enough accuracy for people not to be poisoned. 

Some of the most interesting features in this dataset are: 
- odor - This is a very easy one to identify for most people
- cap-shape - This is usually easy to look at
- cap-surface - This is something you can both feel and see


```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import zscore
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix, precision_recall_curve
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import confusion_matrix
from sklearn.naive_bayes import GaussianNB
from inspect import signature
from warnings import simplefilter
simplefilter(action='ignore', category=FutureWarning)
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz
import graphviz
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
sns.set()
from sklearn.metrics import f1_score

import warnings
warnings.filterwarnings('ignore')
```


```python
df = pd.read_csv("https://raw.githubusercontent.com/ThoMot/DataHost/master/DataScience2019/mushrooms.csv")
```

## Data Exploration:

Taking an overall look at the data, it looks like there are no columns with any missing data. The data set consists of 8124 entries, each with attributes distibuted over 23 columns. All of the data are string values, which are categorical in nature. 

- 1. **Class** - What we are predicting, edible or poisonous
- 2. **cap-shape**
    - b: bell, c: conical, x: convex, f: flat, k: knobbed, s: sunken 
- 4. **Cap-surface** - How is the cap built? 
    -  f: fibrous, g: grooves, y: scaly, s: smooth
- 3. **Cap-color**
    - n: brown, b: buff, c: cinnamon, g: gray, r: green, p: pink, u: purple, e: red, w: white, y: yellow
- 4. **bruises** - Does the shroom have bruses on it?
    - true/false
- 5. **odor**
    - a: almond, l: anise, c: creosote, y: fishy, f: foul, m: musty, n: none, p: pungent, s: spicy
- 6. **gill-attachment** - How is the gill attached
    - a: attached, d: descending, f: free, n: notched
- 7. **gill-spacing** - How are the gills spaced out
    - c: close, w: crowded, d: distant
- 8. **gill-size**
    - b: broad, n: narrow
- 9. **gill-color** 
    - k: black, n: brown, b: buff, h: chocolate, g: gray,  r: green, o: orange, p: pink, u: purple, e: red, w: white, y: yellow
- 10. **stalk-shape** 
    - e: enlarging, t: tapering
- 11. **stalk-root** - Shape of the stalk root
    - b: bulbous, c: club, u: cup, e: equal, z: rhizomorphs, r: rooted, ?: missing
- 12. **stalk-surface-above-ring**
    - f: fibrous, y: scaly, k: silky, s: smooth
- 13. **stalk-surface-below-ring** 
    - f: fibrous, y: scaly, k. silky, s: smooth
- 14. **stalk-color-above-ring**
    - n: brown, b:. buff, c: cinnamon, g: gray, o: orange, p:  pink, e: red, w: white, y: yellow
- 15. **stalk-color-below-ring**
    - n: brown, b: buff, c: cinnamon, g: gray, o: orange, p:  pink, e: red, w: white, y: yellow
- 16. **veil-type** 
    - p: partial, u: universal
- 17. **veil-color** 
    - n: brown, o: orange, w: white, y: yellow
- 18. **ring-number** 
    - n: none, o: one, t: two
- 19. **ring-type** 
    - c: cobwebby, e: evanescent, f: flaring, l: large, n: none, p: pendant, s: sheathing, z: zone
- 20. **spore-print-color**
    - k: black n: brown, b: buff, h: chocolate, r: green, o:  orange, u:. purple, w: white, y: yellow
- 21. **population** 
    - a: abundant, c: clustered, n: numerous, s: scattered, v: several, y: solitary
- 22. **habitat**: 
    - g: grasses, l: leaves, m: meadows, p: paths, u: urban, w: waste, d: woods


```python
df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 8124 entries, 0 to 8123
    Data columns (total 23 columns):
    class                       8124 non-null object
    cap-shape                   8124 non-null object
    cap-surface                 8124 non-null object
    cap-color                   8124 non-null object
    bruises                     8124 non-null object
    odor                        8124 non-null object
    gill-attachment             8124 non-null object
    gill-spacing                8124 non-null object
    gill-size                   8124 non-null object
    gill-color                  8124 non-null object
    stalk-shape                 8124 non-null object
    stalk-root                  8124 non-null object
    stalk-surface-above-ring    8124 non-null object
    stalk-surface-below-ring    8124 non-null object
    stalk-color-above-ring      8124 non-null object
    stalk-color-below-ring      8124 non-null object
    veil-type                   8124 non-null object
    veil-color                  8124 non-null object
    ring-number                 8124 non-null object
    ring-type                   8124 non-null object
    spore-print-color           8124 non-null object
    population                  8124 non-null object
    habitat                     8124 non-null object
    dtypes: object(23)
    memory usage: 1.4+ MB


When looking at df.info() it initially looks like there are no missing values. But when looking into the data of the stalk-root feature, we can tell that there are actually quite a few missing values in this column. When graphing this column based on the class of mushroom, edible/poinsonous we can see that a high percentage of this column is missing. This is denoted by a questionmasrk **?** in this dataset. 


```python
df["stalk-root"].value_counts()
```




    b    3776
    ?    2480
    e    1120
    c     556
    r     192
    Name: stalk-root, dtype: int64




```python
plt.figure(figsize=(15,5));
sns.countplot(df["stalk-root"], hue=df["class"]);
plt.title("number of different stalk-roots dependent of class", size="20");
plt.xlabel("Stalk-root shape", size="15");
plt.ylabel("Number of instances", size="15"); 
plt.legend(["poisonous", "edible"], prop={'size': 15});
plt.xticks(size="12");

```


![png](resources/output_10_0.png)


It is important to look into whether or not the dataset is imbalanced. If the data is imbalanced, one class has a lot more instances than the other. The best classifcation situation would be one where there are about equally as many instances of both classes.

The plot below shows the distribution of the two different classes we are going to predict. There seem to be slightly more edible mushrooms than poisonous ones. It seems like we have a fairly balanced dataset.


```python
plt.figure(figsize=(15,5));
sns.countplot(df["class"]);
plt.title("Class distribution", size="20");
plt.xlabel("Class, poisonous, edible", size="15");
plt.ylabel("Number of instances", size="15"); 
plt.xticks(size="12");
```


![png](resources/output_12_0.png)



```python
plt.figure(figsize=(15,5));
sns.countplot(df["odor"], hue=df["class"]);
plt.title("Class distribution", size="20");
plt.xlabel("Class, poisonous, edible", size="15");
plt.ylabel("Number of instances", size="15"); 
plt.xticks(size="12");
```


![png](resources/output_13_0.png)


When looking at the subplots below, we can tell that some columns are very skewed towards one value. 
- veil-type only contains values of type p
- veil-color contains an overwhelming amount of white instances
- gill-attachment contains a very skewed amount of values for free
- ring-number is very skewed towards one ring


```python
df["veil-type"].value_counts()
```




    p    8124
    Name: veil-type, dtype: int64




```python
df["veil-color"].value_counts()
```




    w    7924
    o      96
    n      96
    y       8
    Name: veil-color, dtype: int64




```python
df["gill-attachment"].value_counts()
```




    f    7914
    a     210
    Name: gill-attachment, dtype: int64




```python
df["ring-number"].value_counts()
```




    o    7488
    t     600
    n      36
    Name: ring-number, dtype: int64



The plot below is a summary plot of all the features to see the distribution within each feature. When looking at them closely we can again see that some features are heavily imbalanced or close to uniform. We will consider dropping some of these values.


```python
fig, ax = plt.subplots(11,2, figsize=(15,25))
fig.suptitle("Count plot of all features", size=25, y=0.92)

sns.countplot(df["class"], ax=ax[0,0])
ax[0,0].set_title("count of edible/poisonous", size="15");
ax[0,0].set_xlabel("class", size="12")

sns.countplot(df["cap-shape"], ax=ax[0,1])
ax[0,1].set_title("count of cap-shapes", size="15");
ax[0,1].set_xlabel("cap-shape", size="12")

sns.countplot(df["cap-surface"], ax=ax[1,0])
ax[1,0].set_title("count of cap-surface", size="15");
ax[1,0].set_xlabel("cap-surface", size="12")

sns.countplot(df["cap-color"], ax=ax[1,1])
ax[1,1].set_title("count of cap-color", size="15");
ax[1,1].set_xlabel("cap-color", size="12")

sns.countplot(df["bruises"], ax=ax[2,0])
ax[2,0].set_title("count of bruising", size="15");
ax[2,0].set_xlabel("bruises", size="12")

sns.countplot(df["odor"], ax=ax[2,1])
ax[2,1].set_title("count of odor", size="15");
ax[2,1].set_xlabel("odors", size="12")

sns.countplot(df["gill-attachment"], ax=ax[3,0])
ax[3,0].set_title("count of gill-attachment", size="15");
ax[3,0].set_xlabel("gill-attachment", size="12")

sns.countplot(df["gill-spacing"], ax=ax[3,1])
ax[3,1].set_title("count of gill-spacing", size="15");
ax[3,1].set_xlabel("gill-spacing", size="12")

sns.countplot(df["gill-size"], ax=ax[4,0])
ax[4,0].set_title("count of gill-size", size="15");
ax[4,0].set_xlabel("gill-size", size="14")

sns.countplot(df["gill-color"], ax=ax[4,1])
ax[4,1].set_title("count of gill-color", size="15");
ax[4,1].set_xlabel("gill-colors", size="12")

sns.countplot(df["stalk-shape"], ax=ax[5,0])
ax[5,0].set_title("count of stalk-shape", size="15");
ax[5,0].set_xlabel("stalk-shapes", size="12")

sns.countplot(df["stalk-root"], ax=ax[5,1])
ax[5,1].set_title("count of stalk-root", size="15");
ax[5,1].set_xlabel("stalk-root", size="12")

sns.countplot(df["stalk-surface-above-ring"], ax=ax[6,0])
ax[6,0].set_title("count of stalk-surface-above-ring", size="15");
ax[6,0].set_xlabel("stalk-surface-above-ring", size="12")

sns.countplot(df["stalk-surface-below-ring"], ax=ax[6,1])
ax[6,1].set_title("count of stalk-surface-below-ring", size="15");
ax[6,1].set_xlabel("stalk-surface-below-ring", size="12")

sns.countplot(df["stalk-color-above-ring"], ax=ax[7,0])
ax[7,0].set_title("count of stalk-color-above-ring", size="15");
ax[7,0].set_xlabel("stalk-color-above-ring", size="12")

sns.countplot(df["stalk-color-below-ring"], ax=ax[7,1])
ax[7,1].set_title("count of stalk-color-below-ring", size="15");
ax[7,1].set_xlabel("stalk-color-below-ring", size="12")

sns.countplot(df["veil-type"], ax=ax[8,0])
ax[8,0].set_title("count of veil-type", size="15");
ax[8,0].set_xlabel("veil-type", size="12")

sns.countplot(df["veil-color"], ax=ax[8,1])
ax[8,1].set_title("count of veil-color", size="15");
ax[8,1].set_xlabel("veil-color", size="12");

sns.countplot(df["ring-number"], ax=ax[9,0])
ax[9,0].set_title("count of ring-number", size="15");
ax[9,0].set_xlabel("ring-number", size="12");

sns.countplot(df["ring-type"], ax=ax[9,1])
ax[9,1].set_title("count of ring-type", size="15");
ax[9,1].set_xlabel("ring-type", size="12");

sns.countplot(df["spore-print-color"], ax=ax[10,0])
ax[10,0].set_title("count of spore-print-color", size="15");
ax[10,0].set_xlabel("spore-print-color", size="12");

sns.countplot(df["population"], ax=ax[10,1])
ax[10,1].set_title("count of population", size="15");
ax[10,1].set_xlabel("population", size="12");
plt.subplots_adjust(hspace = 0.75)
```


![png](resources/output_20_0.png)


## Data Cleaning

- Based on the plots we used to explore the data we can tell that stalk-root had a lot of missing values. it will not be very useful in making predictions. This feature will be dropped.
- Veil-type only contains values of type p which means that this column is the same for every instance in the dataset. It is therefore not useful in making predictions and it will be removed. 
- veil-color had very few values of any other type than white
- ring-number had very few values of any other type than one ring
- gill-attachment had very few values of any other type than free


```python
df.drop(columns=["veil-type", "stalk-root", "veil-color", "ring-number", "gill-attachment"], inplace=True)
```

The target variable is set to 0 = poisonous and 1 = edible.


```python
df["class"] = (df["class"] == "e").astype(int)
```

Since all of the data we are working with is categorical we need to get dummies of the dataframe so we can use it in scikit-learn.


```python
columns = df.columns[df.columns != "class"]
df1 = df.drop(columns=["class"])
df1 = pd.get_dummies(df1, columns=columns)
```

The dataframe now has 111 columns/features, that all contain numeric non-categorical data, ready for scikit-learn machine learning.

# Naive Bayes

# Model 1 - Naive Bayes

For the first model we will be using Naive Bayes, since this is a very fast and easy classification model to implement. However we should be careful of strongly correlated values.


```python
X = df1.values
y = df["class"].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
```

#### Accuracy
When looking at the baseline accuracy this seems reasonable from the plot we made in exploration of the class variable, where we saw a distribution of almost 50/50. The accuracy of the model however comes out to 0.981 which is very very high. It seems that the data we are working with gives extremely good predictions when using all the features of the set.


```python
clf = GaussianNB()
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

print("The baseline accuracy is: {:.3}".format(1 - y_train.mean()))
print("The accuracy of the model is: {:.3}".format(clf.score(X_test, y_test)))
```

    The baseline accuracy is: 0.481
    The accuracy of the model is: 0.981



```python
def print_conf_mtx(y_true, y_pred, classes=None):
    """ Print a confusion matrix (two classes only). """
    
    if not classes:
        classes = ['poisonous', ' edible']
   	 
    # formatting
    max_class_len = max([len(s) for s in classes])
    m = max(max_class_len, len('predicted')//2 + 1)
    n = max(len('actual')+1, max_class_len)
    left   	= '{:<10s}'.replace('10',str(n))
    right  	= '{:>10s}'.replace('10',str(m))
    big_center = '{:^20s}'.replace('20',str(m*2))
    
    cm = confusion_matrix(y_test, y_pred)
    print((left+big_center).format('', 'predicted'))
    print((left+right+right).format('actual', classes[0], classes[1]))
    print((left+right+right).format(classes[0], str(cm[0,0]), str(cm[0,1])))
    print((left+right+right).format(classes[1], str(cm[1,0]), str(cm[1,1])))

def getPrecision(y_true, y_pred):
    cm = confusion_matrix(y_test, y_pred)
    pospos = cm[[1],1]
    poscol = cm[[0],1]
    return (pospos / (pospos + poscol))[0]

def getRecall(y_true, y_pred):
    cm = confusion_matrix(y_test, y_pred)
    #print(cm.shape)
    pospos = cm[[1],1]
    posrow = cm[[1],0]
    return (pospos / (pospos + posrow))[0]

def precisionRecallCurve(y_probs, y_test):
    thresholds = np.linspace(0,1,100)
    precision_vals = []
    recall_vals = []
    for t in thresholds:
        y_pred = (y_probs > t).astype(int)
        precision_vals.append(precision_score(y_pred, y_test))
        recall_vals.append(recall_score(y_pred, y_test))
    ticks = np.linspace(0,1,10)
    labels = ticks.round(1)
    precision_vals = np.array(precision_vals)
    recall_vals = np.array(recall_vals)
    precision_vals = np.insert(precision_vals, 0, 1.0)
    recall_vals = np.insert(recall_vals, 0, 0)
    recall_vals[-1] = 1.0
    plt.title("Precision/Recall")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.fill_between(recall_vals, precision_vals, alpha=0.2, color='b')
    plt.plot(recall_vals, precision_vals)

```

#### Confusion Matrix

When looking at the confusion matrix it looks like both the precision and recall values are very good. The precision is better than the recall, which is very important for this kind of classification problem it is very important to get the ones predicted to be edible to actually be edible because the consequences of false positives in this case could be death.


```python
print_conf_mtx(y_test, y_pred)
```

                 predicted     
    actual   poisonous   edible
    poisonous     1180        1
     edible         46     1211


#### Precision/Recall

- Precision: of the positive predictions, what fraction are correct
- Recall: of the positive cases, which fraction are predicted positive

We definitely want a higher precision than recall here even though there in this canse only is a very small difference.


```python
print("Precision: {:.2f}".format(getPrecision(y_test, y_pred)))
print("Recall: {:.2f}".format(getRecall(y_test, y_pred)))
```

    Precision: 1.00
    Recall: 0.96



```python
te_errs = []
tr_errs = []
tr_sizes = np.linspace(100, X_train.shape[0], 10).astype(int)
for tr_size in tr_sizes:
  X_train1 = X_train[:tr_size,:]
  y_train1 = y_train[:tr_size]
  
  clf.fit(X_train1, y_train1)

  tr_predicted = clf.predict(X_train1)
  err = (tr_predicted != y_train1).mean()
  tr_errs.append(err)
  
  te_predicted = clf.predict(X_test)
  err = (te_predicted != y_test).mean()
  te_errs.append(err)
```

#### Learning Curve

Looking at the learning cure we can tell that the error is very small, even at the beginning of the graph. And the graphs are following each other very very closely.


```python
plt.figure(figsize=(10,5))
sns.lineplot(x=tr_sizes, y=te_errs);
sns.lineplot(x=tr_sizes, y=tr_errs);
plt.title("Learning curve Model 1", size=15);
plt.xlabel("Training set size", size=15);
plt.ylabel("Error", size=15);
plt.legend(['Test Data', 'Training Data']);
```


![png](resources/output_42_0.png)


#### Conclusion Model 1

This model has a higher precision than recall which is very useful in this type of classification. We would much rather a person not eat an edible mushroom than them eating something poisonous. The accuracy is also very good.

# Model 2 - Naive Bayes with Feature Selection

#### Feature Selection:

For the second model we will do feature selection to try to see if we can get the same accuracy as in the previous model with fewer features. When letting the algorithm pick the best 10 columns, we can see that we get an accuracy of 1 at feature number 6.


```python
X = df1.values
y = df["class"].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
clf2 = GaussianNB()
clf2.fit(X_train, y_train);


def bestFeatures(num):
    remaining = list(range(X_train.shape[1]))
    selected = []
    n = num
    while len(selected) < n:
        min_acc = -1e7
        for i in remaining:
            X_i = X_train[:,selected+[i]]
            scores = cross_val_score(GaussianNB(), X_i, y_train,
           scoring='accuracy', cv=3)
            accuracy = scores.mean() 
            if accuracy > min_acc:
                min_acc = accuracy
                i_min = i

        remaining.remove(i_min)
        selected.append(i_min)
        print('num features: {}; accuracy: {:.2f}'.format(len(selected), min_acc))
    return selected
```


```python
selected = df1.columns[bestFeatures(10)]
```

    num features: 1; accuracy: 0.89
    num features: 2; accuracy: 0.94
    num features: 3; accuracy: 0.96
    num features: 4; accuracy: 0.96
    num features: 5; accuracy: 0.99
    num features: 6; accuracy: 1.00
    num features: 7; accuracy: 1.00
    num features: 8; accuracy: 1.00
    num features: 9; accuracy: 1.00
    num features: 10; accuracy: 1.00


Here we see that the best features picked by the algorithm are to do with odor, habitat and spore print color. These are features that might be easy for people to identify, however spore prints might not always be present and stalk color below the ring can also be hard to distinguish. We will explore if we can find such features that are easier to observe and give good accuracy in the next model.


```python
predictors = selected[0:6].values
print("The best 6 features are:", predictors)
```

    The best 6 features are: ['odor_n' 'odor_a' 'habitat_m' 'spore-print-color_r' 'odor_l'
     'stalk-color-below-ring_y']


#### Correlation Plot of chosen features

This heatmap shows the correlation between the features chosen in feature selection. Naive Bayes models don't do well with strongly correlated features, in this plot we can tell that habitat seems to be somewhat correlated with odors almond and anise but the model does not seem to suffer from this.


```python
plt.figure(figsize=(10,5))
corr = df1[predictors].corr();
sns.heatmap(corr);
plt.title("Heatmap of features used in model 2");
```


![png](resources/output_52_0.png)


Training our new model using the predictors found by feature selection


```python
X = df1[predictors].values
y = df["class"].values

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.3, random_state=42)

clf2 = GaussianNB();
clf2.fit(X_train, y_train);
```

#### Accuracy
Looking at the accuracy for this model it has gone up by 0.01 percent using these six predictors. This is a very good model for predicting edible mushrooms.


```python
y_pred = clf2.predict(X_test)
print("The baseline accuracy is: {:.3}".format(1 - y_train.mean()))
print("The accuracy of the model is: {:.3}".format(clf2.score(X_test, y_test)))
```

    The baseline accuracy is: 0.481
    The accuracy of the model is: 0.996


#### Confusion Matrix:

Looking at the confusion Matrix for the model we can tell that the results have changed ever so slightly. There is now a better Recall value while Precision has gone a bit down. One could argue that this development is not favourable since it is very bad to predict a mushroom to be edible and it ending up actually being poisonous.


```python
print_conf_mtx(y_test, y_pred)
```

                 predicted     
    actual   poisonous   edible
    poisonous     1172        9
     edible          0     1257


#### Precision/Recall:

The precision recall values are confirming what we saw in the confusion matrix, that Recall is now perfect while Precision has gone down, even if only by a little.


```python
print("Precision: {:.3f}".format(getPrecision(y_test, y_pred)))
print("Recall: {:.3f}".format(getRecall(y_test, y_pred)))
```

    Precision: 0.993
    Recall: 1.000


#### Learning Curve:

The learning curve looks very interesting. We can tell that the error is very very small. We do not seem to have a high variance case since the training error is so low and the gap between the curves is pretty small. It does not look like a high bias problem either since the training error is so low. It looks like the model is fitted very well to the training data and that it works very well for the test data.


```python
te_errs = []
tr_errs = []
tr_sizes = np.linspace(100, X_train.shape[0], 10).astype(int)
for tr_size in tr_sizes:
  X_train1 = X_train[:tr_size,:]
  y_train1 = y_train[:tr_size]
  
  clf2.fit(X_train1, y_train1)

  tr_predicted = clf2.predict(X_train1)
  err = (tr_predicted != y_train1).mean()
  tr_errs.append(err)
  
  te_predicted = clf2.predict(X_test)
  err = (te_predicted != y_test).mean()
  te_errs.append(err)
```


```python
plt.figure(figsize=(10,5))
sns.lineplot(x=tr_sizes, y=te_errs);
sns.lineplot(x=tr_sizes, y=tr_errs);
plt.title("Learning curve Model 2", size=15);
plt.xlabel("Training set size", size=15);
plt.ylabel("Error", size=15);
plt.legend(['Test Data', 'Training Data']);
```


![png](resources/output_63_0.png)


#### Conclusion Model 2
Overall accuracy is very good for this model, it seems doing forward feature selection from all the columns in the data set is very good. We get a very high precision and recall as well. However some of the features selected by forward feature selection can be hard to identify while out looking for mushrooms.

# Model 3 - Naive Bayes - Making the most usable prediction model

In this model we will try to make to most usable prediction model possible by selecting features that are easy for people going shrooming to distiguish between. The features we selected are: 
- cap-surface
- odor 
- cap-shape
These are all features that should be easily distinguishable when going out looking for mushrooms and it should be easy for people to take note of these characteristics. We are getting dummy variables for these featues as they are categorical in nature.


```python
cols = ["cap-surface", "odor", "cap-shape"]
df3 = df[cols]
df3 = pd.get_dummies(df3)
```


```python
X = df3.values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
clf3 = GaussianNB()
clf3.fit(X_train, y_train);
```

#### Forward feature selection: 

When doing forward feature selection based on the features we chose for this model, we get an accuracy of 99 when we get to 5 features.


```python
selected = df3.columns[bestFeatures(5)]
```

    num features: 1; accuracy: 0.89
    num features: 2; accuracy: 0.94
    num features: 3; accuracy: 0.96
    num features: 4; accuracy: 0.96
    num features: 5; accuracy: 0.99


The best five features to use for these predictions according to forward feature selection are: 
- odor none
- odor almond
- cap shape bell
- cap shape conical
- odor anise


```python
predictors3 = selected[0:10].values
print("The best 5 features are:", predictors3)
```

    The best 5 features are: ['odor_n' 'odor_a' 'cap-shape_b' 'cap-shape_c' 'odor_l']


#### Correlation Plot

It does not seem like any of the features used in this model are strongly correlated. This is very good for Naive bayes predictions.


```python
plt.figure(figsize=(10,5))
corr3 = df3[predictors3].corr();
sns.heatmap(corr3);
plt.title("Heatmap of features used in model2");
```


![png](resources/output_74_0.png)


#### Accuracy

The accuracy of the model is high at 0.985. This isn't a perfect model like we almost acheived with forward feature selection in the previous model, but it is still high enough to be reasonably good at predicting edible mushrooms.


```python
y_pred = clf3.predict(X_test)
print("The baseline accuracy is: {:.3}".format(1 - y_train.mean()))
print("The accuracy of the model is: {:.3}".format(clf3.score(X_test, y_test)))
```

    The baseline accuracy is: 0.481
    The accuracy of the model is: 0.985


#### Confusion Matrix

When looking at the confusion matrix produced for this model we can tell that the recall is perfect while the precision is a little bit off. It is not much but there are still 36 mushrooms that will be classified aas edible that are actually poisonous. This is not a good outcome. However it is a very very small fraction. One could argue that is would be very important to get precision to 1 here, because you would not want people believing in your model and then ending up being severely poisoned. 


```python
print_conf_mtx(y_test, y_pred)
```

                 predicted     
    actual   poisonous   edible
    poisonous     1145       36
     edible          0     1257


#### Precision/ Recall

Like stated in the previous paragraph we can tell that Recall is perfect and we have a little bit to go on when it comes to precision. Hwever, precision for this classification is still really good at 0.97.


```python
print("Precision: {:.3f}".format(getPrecision(y_test, y_pred)))
print("Recall: {:.3f}".format(getRecall(y_test, y_pred)))
```

    Precision: 0.972
    Recall: 1.000



```python
te_errs = []
tr_errs = []
tr_sizes = np.linspace(100, X_train.shape[0], 10).astype(int)
for tr_size in tr_sizes:
  X_train1 = X_train[:tr_size,:]
  y_train1 = y_train[:tr_size]
  
  clf3.fit(X_train1, y_train1)

  tr_predicted = clf3.predict(X_train1)
  err = (tr_predicted != y_train1).mean()
  tr_errs.append(err)
  
  te_predicted = clf3.predict(X_test)
  err = (te_predicted != y_test).mean()
  te_errs.append(err)
```

#### Learning Curve

This learning curve also shows us a very small error. However, it does look like we get a very low error for the training data that actaully goes back up a bit when getting to around 1500 in training size. We also have a point where the two curves intersect at right before 1000 instances. The two lines do however smooth out and stay fairly close to each other as more data is provided. It looks like it is stabilizing at around an error of 0.0150.


```python
plt.figure(figsize=(10,5))
sns.lineplot(x=tr_sizes, y=te_errs);
sns.lineplot(x=tr_sizes, y=tr_errs);
plt.title("Learning curve Model 1", size=15);
plt.xlabel("Training set size", size=15);
plt.ylabel("Error", size=15);
plt.legend(['Test Data', 'Training Data']);
```


![png](resources/output_83_0.png)


#### Conclusion Model 3: 

It looks like overall we ended up with a marginally lower accuracy and a little bit of a lower precision for this model than for the last model using feature selection on all features in the dataset. However this might be a more usable model for poeple actually in the woods looking for mushrooms. The features we worked with in this case are easier to distinguish than the ones found in the previous even more accurate model.

# Tree Classifiers

# Model 4 - Tree Classification - All Features

For the fourth model we are using a tree classifier to compare to our earlier naive bayes models. First we are making a model using all the features in the dataset to see if we get as good of an accuracy here as we did with Naive Bayes.


```python
np.random.seed(42)
```


```python
X = df1.values

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.3, random_state=42)

clf4 = DecisionTreeClassifier(max_depth=3, random_state=0);
clf4.fit(X_train, y_train);
```

#### Tree Graph

A graph displaying our model fit to the training data is produced. From the graph we can tell that our baseline accuracy should be 0.48 since looking at all data points, 48% of the mushrooms are poisonous. So if we haven't looked at the dataset at all before guessing that a mushroom is poisonous, we have a 48% chance of being right.


```python
target_names = ['Poisonous', 'Edible']
dot_data = export_graphviz(clf4, precision=2,
feature_names=df1.columns.values,
proportion=True,
class_names=target_names,
filled=True, rounded=True,
special_characters=True)
graph = graphviz.Source(dot_data)
graph
```




![svg](resources/output_91_0.svg)



#### Accuracy: 

The accuracy for this model is the same as we got in the first Naive bayes model. 


```python
print("Baseline Accuracy: {:.3}".format(1-y_train.mean()))
y_predict = clf.predict(X_test)
print("Accuracy of Model: {:.3}".format((y_predict == y_test).mean()))
```

    Baseline Accuracy: 0.481
    Accuracy of Model: 0.981


#### Confusion Matrix:

When looking at the confusion matrix it does look identical to the Naive bayes model we made using all features of the dataset. The difference in the two models when it comes to precision and recall seem to be indistinguishable.


```python
print_conf_mtx(y_test, y_predict)
```

                 predicted     
    actual   poisonous   edible
    poisonous     1180        1
     edible         46     1211


#### Precision/Recall: 

The precision and recall values are good values for this classification problem as we would want the least amount of people to die, using our classifier. In this case 1 person would eat a mushrom wrongly classified as edible when it is not. This still gives us a precision of 1 when rounded to three decimals.


```python
print("Precision: {:.2f}".format(getPrecision(y_test, y_predict)))
print("Recall: {:.2f}".format(getRecall(y_test, y_predict)))
```

    Precision: 1.00
    Recall: 0.96



```python
te_errs = []
tr_errs = []
tr_sizes = np.linspace(100, X_train.shape[0], 10).astype(int)
for tr_size in tr_sizes:
  X_train1 = X_train[:tr_size,:]
  y_train1 = y_train[:tr_size]
  
  clf4.fit(X_train1, y_train1)

  tr_predicted = clf4.predict(X_train1)
  err = (tr_predicted != y_train1).mean()
  tr_errs.append(err)
  
  te_predicted = clf4.predict(X_test)
  err = (te_predicted != y_test).mean()
  te_errs.append(err)
```

#### Learning Curve: 

The learning curve is what is the most different from our first naive bayes model using all features. 
It looks like the error starts out low and then increases as mode data is provided. This is very surprising. However, when you look at the actual error score, it is still very very small.


```python
plt.figure(figsize=(10,5))
sns.lineplot(x=tr_sizes, y=te_errs);
sns.lineplot(x=tr_sizes, y=tr_errs);
plt.title("Learning curve Model 4", size=15);
plt.xlabel("Training set size", size=15);
plt.ylabel("Error", size=15);
plt.legend(['Test Data', 'Training Data']);
```


![png](resources/output_100_0.png)


#### Conclusion Model 5: 

It seems like the tree classifier does just as well as the bayes model when using all the features of the dataset. The main difference between the two models is the learning cure and how the classifiers develop when getting more data to practice and test on. In addition this model has perfect precision which is somehting very useful in this classification problem.

# Model 5 - Tree Classifier - The Most Usable prediction model

Earlier we defined a few features that were easy for people to identify when mushroom hunting and that gave a good enough accuracy to where it would be useful. We are now going to take a look at how a tree classifier treats the same features.
The features in question are: 
- cap-surface
- odor
- cap-shape


```python
X = df3.values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
```


```python
def bestTreeFeatures(num):
    remaining = list(range(X_train.shape[1]))
    selected = []
    n = num
    while len(selected) < n:
        min_acc = -1e7
        for i in remaining:
            X_i = X_train[:,selected+[i]]
            scores = cross_val_score(DecisionTreeClassifier(max_depth=5, random_state=42), X_i, y_train,
           scoring='accuracy', cv=3)
            accuracy = scores.mean() 
            if accuracy > min_acc:
                min_acc = accuracy
                i_min = i

        remaining.remove(i_min)
        selected.append(i_min)
        print('num features: {}; accuracy: {:.2f}'.format(len(selected), min_acc))
    return selected
```

#### Forward feature selection

When doing forward feature slection using the same 3 features made into dummy variables we can tell that this model actually reaches an accuracy of 0.99 at three features, as opposed to our previous model that needed 5 features. 


```python
selected = df3.columns[bestTreeFeatures(5)]
```

    num features: 1; accuracy: 0.89
    num features: 2; accuracy: 0.94
    num features: 3; accuracy: 0.99
    num features: 4; accuracy: 0.99
    num features: 5; accuracy: 0.99


Looking at the three best columns we can tell that this classifier actually gets a 0.99 accuracy only using odor, which is only one feature from our original dataframe. Now made into three columns using dummy variables. This is really interestng. Maybe people can distingush edible mushrooms based only on their smell.
- odor none
- odor almond
- odor anis


```python
predictors = selected[0:3].values
print("The best 3 features are:", predictors)
```

    The best 3 features are: ['odor_n' 'odor_a' 'odor_l']


#### Tree graph

The tree we produce from fitting our model to the trainig data. The max depth is set to 3, and our baseline accuracy is once again 48 percent when predicting that a mushroom is poisonous. At the leaf nodes there is still a good amount of samples left. 


```python
X = df1[predictors].values

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.3, random_state=42)

clf5 = DecisionTreeClassifier(max_depth=3, random_state=0);
clf5.fit(X_train, y_train);

target_names = ['Poisonous', 'Edible']
dot_data = export_graphviz(clf5, precision=2,
feature_names=df1[predictors].columns.values,
proportion=True,
class_names=target_names,
filled=True, rounded=True,
special_characters=True)
graph = graphviz.Source(dot_data)
graph
```




![svg](resources/output_111_0.svg)



#### Accuracy 

The accuracy of this model is 0.001 worse than the previous model we used for predicting on these columns. The accuracy is however still very high.


```python
print("Baseline Accuracy: {:.3f}".format(1-y_train.mean()))
y_predict5 = clf5.predict(X_test)
print("Model 5 Accuracy: {:.3f}".format((y_predict5 == y_test).mean()))
```

    Baseline Accuracy: 0.481
    Model 5 Accuracy: 0.984


#### Confusion Matrix

In this confusion matrix we have two more instances that were predicted edible that are in fac poisonous. We can expect precision to be marginally lower in this model than in the Naive Bayes model. Recall is still at 1. 


```python
print_conf_mtx(y_test, y_predict5)
```

                 predicted     
    actual   poisonous   edible
    poisonous     1142       39
     edible          0     1257


#### Precision/Recall
We can tell that precision has decreased by 0.002 points in this model. This is not a favorable outcome as this would mean two more people could potentially get sick from eating a mushroom that is predicted edible and was not. 


```python
print("Precision: {:.3f}".format(getPrecision(y_test, y_predict5)))
print("Recall: {:.3f}".format(getRecall(y_test, y_predict5)))
```

    Precision: 0.970
    Recall: 1.000


#### Precision Recall Curve

The following precision/recall curve is close to a perfect curve which supports the confusion matrix we got for this model. We can see that there is a drop in precision when recall goes from 0.9 to 1.0.


```python
y_probs = clf5.predict_proba(X_test)
y_probs = y_probs[:,1]
precisionRecallCurve(y_probs, y_test)
```


![png](resources/output_119_0.png)


#### ROC curve

By looking at the ROC curve, we can see the same trend from the precision/recall curve that the model predicts some false positive while true positive is nearly perfect with an AUC rating of 0.99.


```python
auc = roc_auc_score(y_test, y_probs)
fpr, tpr, thresholds = roc_curve(y_test, y_probs)
plt.plot(fpr, tpr, color='red', label='ROC')
plt.plot([0, 1], [0, 1], color='black', linestyle='--')
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.title('ROC curve for Edible Mushrooms')
plt.legend()
plt.show()
print('AUC: %.2f' % auc)
```


![png](resources/output_121_0.png)


    AUC: 0.99


#### Learning Curve



```python
te_errs = []
tr_errs = []
tr_sizes = np.linspace(100, X_train.shape[0], 10).astype(int)
for tr_size in tr_sizes:
  X_train1 = X_train[:tr_size,:]
  y_train1 = y_train[:tr_size]
  clf5.fit(X_train1, y_train1)
  tr_predicted = clf5.predict(X_train1)
  err = (tr_predicted != y_train1).mean()
  tr_errs.append(err)
  te_predicted = clf5.predict(X_test)
  err = (te_predicted != y_test).mean()
  te_errs.append(err)
```

#### Learning curve

This learning curve is a bit surprising when looking at the test data. The error seems to be pretty much consitent through out the whole progress. 


```python
plt.figure(figsize=(10,5))
sns.lineplot(x=tr_sizes, y=te_errs);
sns.lineplot(x=tr_sizes, y=tr_errs);
plt.title("Learning curve Model 5", size=15);
plt.xlabel("Training set size", size=15);
plt.ylabel("Error", size=15);
plt.legend(['Test Data', 'Training Data']);
```


![png](resources/output_125_0.png)


#### Feature Importance

Looking at the feature importances, it seems like the most important feature for this model is no odor. The other ones seem less significant. 


```python
plt.figure(figsize=(10,5))
sns.barplot(x=clf5.feature_importances_, y=df1[predictors].columns);
plt.title("Feature Importance model 5", size=15);
plt.xlabel("Importance", size=15);
plt.ylabel("Features", size=15);
```


![png](resources/output_127_0.png)


#### Conclusion Model 5: 

We saw an equal accuracy in feature selection as in the previous one using only 3 features instead of 5. However the actual accuracy when making predictions seem to be marginally lower than the Naive bayes model. The precision value decreased a ittle, so 2 more mushrooms that are correctly classified in the previous model are wrondly classified as edible in this classifier.

# Logistic Regression

# Model 6 - Logistic Regression - All Features

For the sixth and seventh model we are going to look into Logistic regression nd how it compares to the other two types of models, Naive bayes and Tree Classifier.

#### Accuracy

The baseline accuracy is at 0.48 which means that for half of the predictions there is a close to 50% chance to predict correct.

After training the model with all the features, we get 100% accuracy. Like we have seen in the other models it seems we have features in the data that makes it easy to predict on all features. This is however a slightly better accuracy than what we have seen for all features in the previous models using all features.


```python
X = df1.values
y = df["class"].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

log = LogisticRegression()
log.fit(X_train, y_train)
print("Baseline accuracy: {:.2f}".format(1-y_train.mean()))
print("Logistic regression accuracy: {:.2}".format(log.score(X_test, y_test)))
```

    Baseline accuracy: 0.48
    Logistic regression accuracy: 1.0


#### Confusion Matrix

Looking at these values this model has very good precision and recall. only one single person will potentially get sick from eating a poisonous mushroom that was classified as edible. This of course is very unfortunate for this person, but it seems like this is the best case scenario for out prediction models.  


```python
y_pred = log.predict(X_test)
print_conf_mtx(y_test, y_pred)
```

                 predicted     
    actual   poisonous   edible
    poisonous     1180        1
     edible          0     1257


#### Precision / Recall

When looking at the precision and recall values for this model, we can see that when rounding to three numbers we get a value of 1 for both precision and recall which is very very good. 


```python
print("Precision: {:.4f}".format(getPrecision(y_test, y_pred)))
print("Recall: {:.4f}".format(getRecall(y_test, y_pred)))
```

    Precision: 0.9992
    Recall: 1.0000



```python
print("F1 score: {:.4f}".format(f1_score(y_test, y_pred)))
```

    F1 score: 0.9996


#### Precision / Recall Curve

The precision recall curve of this model looks very good. And that goes well with the accuracy we saw when calculation it earlier and the precision recall values. It shows us that we can accheive a 1 for recall as well as for precision at the same time which is what we acheived with the model.


```python
y_probs = log.predict_proba(X_test)
y_probs = y_probs[:,1]
precisionRecallCurve(y_probs, y_test)
```


![png](resources/output_140_0.png)


#### ROC curve

The ROC curve is a probability curve for different classes. ROC tells us how good the model is for classifying in these classes. The ROC curve has False Positive Rate on the X-axis and True Positive Rate on the Y axis. The area between the ROC curve and the axis is AUC. A bigger area signifies a better model for distinguishing between classes. The best posible value for ROC is 1, that would mean it should hug the top left corner of the graph. 
Looking at the curve below, it seems like the model is doing very well. The AUC is 1 which is very very good.


```python
auc = roc_auc_score(y_test, y_probs)
fpr, tpr, thresholds = roc_curve(y_test, y_probs)
plt.plot(fpr, tpr, color='red', label='ROC')
plt.plot([0, 1], [0, 1], color='black', linestyle='--')
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.title('ROC curve for Model 6 All features')
plt.legend()
plt.show()
print('AUC: %.2f' % auc)
```


![png](resources/output_142_0.png)


    AUC: 1.00



```python
te_errs = []
tr_errs = []
tr_sizes = np.linspace(100, X_train.shape[0], 10).astype(int)
for tr_size in tr_sizes:
  X_train1 = X_train[:tr_size,:]
  y_train1 = y_train[:tr_size]
  
  log.fit(X_train1, y_train1)

  tr_predicted = log.predict(X_train1)
  err = (tr_predicted != y_train1).mean()
  tr_errs.append(err)
  
  te_predicted = log.predict(X_test)
  err = (te_predicted != y_test).mean()
  te_errs.append(err)
```

#### Learning Curve

When looking at the plot below for the learning curves it initially looks like there is a big gap between the curves, but looking at the error values it is evident that these curves are actually very close together. 


```python
plt.figure(figsize=(10,5))
sns.lineplot(x=tr_sizes, y=te_errs);
sns.lineplot(x=tr_sizes, y=tr_errs);
plt.title("Learning curve Model 6", size=15);
plt.xlabel("Training set size", size=15);
plt.ylabel("Error", size=15);
plt.legend(['Test Data', 'Training Data']);
```


![png](resources/output_145_0.png)


#### Conclusion Model 6: 

This model had a better accuracy than the previous models we have made with all features from the data set. However the precision was a little bit lower. When it comes to the learning curve it seems to have both the sets very close together in error and they do not really move toward or away from each other. 

# Model 7 - Logistic Regression - The Most Usable Prediction Model

In this model we are looking to make a useful predictor using predictors that are easy for people to identify. We will be using Logistic regression and compare it to the other models we have made so far.


```python
X = df3.values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
```


```python
def bestLogFeatures(num):
    remaining = list(range(X_train.shape[1]))
    selected = []
    n = num
    while len(selected) < n:
        min_acc = -1e7
        for i in remaining:
            X_i = X_train[:,selected+[i]]
            scores = cross_val_score(LogisticRegression(), X_i, y_train,
           scoring='accuracy', cv=3)
            accuracy = scores.mean() 
            if accuracy > min_acc:
                min_acc = accuracy
                i_min = i

        remaining.remove(i_min)
        selected.append(i_min)
        print('num features: {}; accuracy: {:.2f}'.format(len(selected), min_acc))
    return selected
```

#### Forward Feature Selection

We are continuing to use the same features we have used to make a useful predictor earlier. That is cap surface, odor, and cap shape. Looking at the output from forward feature selection we can tell that we hit a cap of 0.99 after 3 features are chosen.


```python
selected = df3.columns[bestLogFeatures(5)]
```

    num features: 1; accuracy: 0.89
    num features: 2; accuracy: 0.94
    num features: 3; accuracy: 0.99
    num features: 4; accuracy: 0.99
    num features: 5; accuracy: 0.99


The best three features are the same as we found in the second Tree classifier. Odor of almond, anis and no odor at all.


```python
predictors = selected[0:3].values
print("The best 3 features are:", predictors)
```

    The best 3 features are: ['odor_n' 'odor_a' 'odor_l']


#### Accuracy

We get a very high accuracy for this model as well. It seems to be the exact same as we got for the second Tree classifier. 


```python
X = df1[predictors].values
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.3, random_state=42)

log2 = LogisticRegression()
log2.fit(X_train, y_train)
print("Baseline accuracy: {:.3f}".format(1-y_train.mean()))
print("Logistic regression accuracy: {:.3}".format(log2.score(X_test, y_test)))
```

    Baseline accuracy: 0.481
    Logistic regression accuracy: 0.984


#### Confusion Matrix 

The confusion matric is also the same as in the last Tree Classification. There are still 39 mushrooms that are classified as edible, when they are actually poisionous.


```python
y_predict7 = log2.predict(X_test)
print_conf_mtx(y_test, y_predict7)
```

                 predicted     
    actual   poisonous   edible
    poisonous     1142       39
     edible          0     1257


#### Precision/Recall

The precision and recall values reflect what we read off the confusion matrix. The recall comes out to one and the precision is a bit lower than we would like it to be.


```python
print("Precision: {:.3f}".format(getPrecision(y_test, y_predict7)))
print("Recall: {:.3f}".format(getRecall(y_test, y_predict7)))
```

    Precision: 0.970
    Recall: 1.000


#### F1 Score
The F1 score reflects a very high precision and recall. The best value one can obtain here is 1. A score of 0.98 is pretty good.


```python
print("F1 score: {:.4f}".format(f1_score(y_test, y_predict7)))
```

    F1 score: 0.9847


#### ROC curve

In this ROC curve we can tell that the area under the curve comes out to 0.99. This is a very high score and reflects that we have a very high true positive rate and a low false positive rate.


```python
y_probs = log2.predict_proba(X_test)
y_probs = y_probs[:,1]
auc = roc_auc_score(y_test, y_probs)
fpr, tpr, thresholds = roc_curve(y_test, y_probs)
plt.plot(fpr, tpr, color='red', label='ROC')
plt.plot([0, 1], [0, 1], color='black', linestyle='--')
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.title('ROC curve for Model 7 Usable')
plt.legend()
plt.show()
print('AUC: %.2f' % auc)
```


![png](resources/output_164_0.png)


    AUC: 0.99



```python
te_errs = []
tr_errs = []
tr_sizes = np.linspace(100, X_train.shape[0], 10).astype(int)
for tr_size in tr_sizes:
  X_train1 = X_train[:tr_size,:]
  y_train1 = y_train[:tr_size]
  
  log2.fit(X_train1, y_train1)

  tr_predicted = log2.predict(X_train1)
  err = (tr_predicted != y_train1).mean()
  tr_errs.append(err)
  
  te_predicted = log2.predict(X_test)
  err = (te_predicted != y_test).mean()
  te_errs.append(err)
```

#### Learning Curve

This learning curve reflects that the error for the test data decreases rapidly as well as the error for the training data. at around a 1000 instances, the curves flatten out and the error stays relatively stable. The area between them is also very small. Looking at the values of error, these errors are marginal.


```python
plt.figure(figsize=(10,5))
sns.lineplot(x=tr_sizes, y=te_errs);
sns.lineplot(x=tr_sizes, y=tr_errs);
plt.title("Learning curve Model 7", size=15);
plt.xlabel("Training set size", size=15);
plt.ylabel("Error", size=15);
plt.legend(['Test Data', 'Training Data']);
```


![png](resources/output_167_0.png)


#### Conclusion Model 7

Overall in this model we can tell that it has the same performance as the last Tree Classifier we built using the same features. Using feature selecton the same features were pickedd out in the same amount of steps. The biggest difference between these models lie in the learning curves in which the Tree classifier seemed to have a constant error for the training set while the learning curve in this logistic regression model seems to have the training and test data error follow each other more closely.

---

# Conclusion

In total we have made 7 models, 3 Naive bayes models, 2 Tree classifiers and 2 Logistic regression models. The first observation we made is that our dataset inherently gives us a very high accuracy and precision/recall values for all of our models. At first we thought this could be because we had an imbalanced dataset but when we went back and looked at the values we are classifying by, Edible/poisonous, there is about 50% of each present in the dataset. 


The best accuracy we achieved was using Logistic regression on all features of the dataset. This gave us an accuracy of 100%. When it comes to making a most useful classifier that uses features that people could easily observe, we got predictions using the fewest features (only three variants of odor) when classifying using Logistic regression and Tree Classification. 

However, the accuracy was actually a little bit higher for our Naive Bayes model using 5 features. The highest accuracy achieved for the "Useful" model, was 0.985. This gave us a precision of 0.972. This means that worst case scenario, 36 people who had their mushroom predicted to be edible, would actually eat a poisonous one. 
The best precision vs recall was obtained using all features and Naive Bayes in model 1.

Overall, based on this dataset, it looks like it is fairly easy, to decide if a mushroom is edible, based on cap-shape and smell. The best predictors are smell of nothing, almond or anis and a cap shape of conical or bell. Still, there is no 100% guarantee that you won't be poisoned using these prediction models.
