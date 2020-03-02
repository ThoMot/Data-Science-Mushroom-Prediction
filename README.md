{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# What is the danger of shrooming? "
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "- **Thora Mothes**\n",
    "- **Mathias Ahrn**"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Introduction:\n",
    "\n",
    "Shrooming refers to the act of looking for mushrooms or mushroom hunting. In this report predictions will be made as to how likely it is for a mushroom with certain characteristics to be poisonous or edible.\n",
    "\n",
    "##### Dataset:\n",
    "The chosen dataset for this report is called **<a href=\"https://archive.ics.uci.edu/ml/datasets/Mushroom\">Mushrooms</a>**, and is a dataset consisting of descriptions of hypothetical samples corresponding to 23 species of mushroom in the Agaricus amd Lepiota Family. The data originally stems from the Audubon Society Field Guide to North American Mushrooms published in 1981, and was donated to <a href=\"https://archive.ics.uci.edu/ml/index.php\">The UCI repository</a> in 1987. \n",
    "\n",
    "The dataset used in this report was downloaded from <a href=\"https://www.kaggle.com/uciml/mushroom-classification\">Kaggle</a> as the UCI repository could not provide a CSV file. However the data from Kaggle has not been modified in any other way than formatting it into a CSV file. \n",
    "\n",
    "- Date downloaded: **12.09.2019**\n",
    "- Hosted: **Thora Mothes GitHub Repo**\n",
    " \n",
    "\n",
    "##### Goal:\n",
    "In this report the goal is to predict whether or not a mushroom is edible. The feature used for output values is called \"Class\" and consists of values **e** and **p**, e for edible, and p for poisonous.\n",
    "\n",
    "##### Features: \n",
    "\n",
    "When it comes to the features we will use for our models we will have two main criteria, we will try to find features that are easy for people to identify when they are looking for mushrooms and still give good enough accuracy for people not to be poisoned. \n",
    "\n",
    "Some of the most interesting features in this dataset are: \n",
    "- odor - This is a very easy one to identify for most people\n",
    "- cap-color - This is also a very easy identifier\n",
    "- "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 1,
   "metadata": {},
   "outputs": [],
   "source": [
    "import pandas as pd\n",
    "import numpy as np\n",
    "import matplotlib.pyplot as plt\n",
    "import seaborn as sns\n",
    "from scipy.stats import zscore\n",
    "from sklearn.linear_model import LogisticRegression\n",
    "from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix, precision_recall_curve\n",
    "from sklearn.model_selection import train_test_split, cross_val_score\n",
    "from sklearn.metrics import confusion_matrix\n",
    "from sklearn.naive_bayes import GaussianNB\n",
    "from inspect import signature\n",
    "from warnings import simplefilter\n",
    "simplefilter(action='ignore', category=FutureWarning)\n",
    "from sklearn.tree import DecisionTreeClassifier\n",
    "from sklearn.tree import export_graphviz\n",
    "import graphviz\n",
    "from sklearn.metrics import precision_score\n",
    "from sklearn.metrics import recall_score\n",
    "sns.set()\n",
    "from sklearn.metrics import f1_score"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "metadata": {},
   "outputs": [],
   "source": [
    "df = pd.read_csv(\"https://raw.githubusercontent.com/ThoMot/DataHost/master/DataScience2019/mushrooms.csv\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Data Exploration:\n",
    "\n",
    "Taking an overall look at the data, it looks like there are no columns with any missing data. The data set consists of 8124 entries, each with attributes distibuted over 23 columns. All of the data are string values, which are categorical in nature. \n",
    "\n",
    "- 1. **Class** - What we are predicting, edible or poisonous\n",
    "- 2. **cap-shape**\n",
    "    - b: bell, c: conical, x: convex, f: flat, k: knobbed, s: sunken \n",
    "- 4. **Cap-surface** - How is the cap built? \n",
    "    -  f: fibrous, g: grooves, y: scaly, s: smooth\n",
    "- 3. **Cap-color**\n",
    "    - n: brown, b: buff, c: cinnamon, g: gray, r: green, p: pink, u: purple, e: red, w: white, y: yellow\n",
    "- 4. **bruises** - Does the shroom have bruses on it?\n",
    "    - true/false\n",
    "- 5. **odor**\n",
    "    - a: almond, l: anise, c: creosote, y: fishy, f: foul, m: musty, n: none, p: pungent, s: spicy\n",
    "- 6. **gill-attachment** - How is the gill attached\n",
    "    - a: attached, d: descending, f: free, n: notched\n",
    "- 7. **gill-spacing** - How are the gills spaced out\n",
    "    - c: close, w: crowded, d: distant\n",
    "- 8. **gill-size**\n",
    "    - b: broad, n: narrow\n",
    "- 9. **gill-color** \n",
    "    - k: black, n: brown, b: buff, h: chocolate, g: gray,  r: green, o: orange, p: pink, u: purple, e: red, w: white, y: yellow\n",
    "- 10. **stalk-shape** \n",
    "    - e: enlarging, t: tapering\n",
    "- 11. **stalk-root** - Shape of the stalk root\n",
    "    - b: bulbous, c: club, u: cup, e: equal, z: rhizomorphs, r: rooted, ?: missing\n",
    "- 12. **stalk-surface-above-ring**\n",
    "    - f: fibrous, y: scaly, k: silky, s: smooth\n",
    "- 13. **stalk-surface-below-ring** \n",
    "    - f: fibrous, y: scaly, k. silky, s: smooth\n",
    "- 14. **stalk-color-above-ring**\n",
    "    - n: brown, b:. buff, c: cinnamon, g: gray, o: orange, p:  pink, e: red, w: white, y: yellow\n",
    "- 15. **stalk-color-below-ring**\n",
    "    - n: brown, b: buff, c: cinnamon, g: gray, o: orange, p:  pink, e: red, w: white, y: yellow\n",
    "- 16. **veil-type** \n",
    "    - p: partial, u: universal\n",
    "- 17. **veil-color** \n",
    "    - n: brown, o: orange, w: white, y: yellow\n",
    "- 18. **ring-number** \n",
    "    - n: none, o: one, t: two\n",
    "- 19. **ring-type** \n",
    "    - c: cobwebby, e: evanescent, f: flaring, l: large, n: none, p: pendant, s: sheathing, z: zone\n",
    "- 20. **spore-print-color**\n",
    "    - k: black n: brown, b: buff, h: chocolate, r: green, o:  orange, u:. purple, w: white, y: yellow\n",
    "- 21. **population** \n",
    "    - a: abundant, c: clustered, n: numerous, s: scattered, v: several, y: solitary\n",
    "- 22. **habitat**: \n",
    "    - g: grasses, l: leaves, m: meadows, p: paths, u: urban, w: waste, d: woods"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "<class 'pandas.core.frame.DataFrame'>\n",
      "RangeIndex: 8124 entries, 0 to 8123\n",
      "Data columns (total 23 columns):\n",
      "class                       8124 non-null object\n",
      "cap-shape                   8124 non-null object\n",
      "cap-surface                 8124 non-null object\n",
      "cap-color                   8124 non-null object\n",
      "bruises                     8124 non-null object\n",
      "odor                        8124 non-null object\n",
      "gill-attachment             8124 non-null object\n",
      "gill-spacing                8124 non-null object\n",
      "gill-size                   8124 non-null object\n",
      "gill-color                  8124 non-null object\n",
      "stalk-shape                 8124 non-null object\n",
      "stalk-root                  8124 non-null object\n",
      "stalk-surface-above-ring    8124 non-null object\n",
      "stalk-surface-below-ring    8124 non-null object\n",
      "stalk-color-above-ring      8124 non-null object\n",
      "stalk-color-below-ring      8124 non-null object\n",
      "veil-type                   8124 non-null object\n",
      "veil-color                  8124 non-null object\n",
      "ring-number                 8124 non-null object\n",
      "ring-type                   8124 non-null object\n",
      "spore-print-color           8124 non-null object\n",
      "population                  8124 non-null object\n",
      "habitat                     8124 non-null object\n",
      "dtypes: object(23)\n",
      "memory usage: 1.4+ MB\n"
     ]
    }
   ],
   "source": [
    "df.info()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "When looking at df.info() it initially looks like there are no missing values. But when looking into the data of the stalk-root feature, we can tell that there are actually quite a few missing values in this column. When graphing this column based on the class of mushroom, edible/poinsonous we can see that a high percentage of this column is missing. This is denoted by a questionmasrk **?** in this dataset. "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "b    3776\n",
       "?    2480\n",
       "e    1120\n",
       "c     556\n",
       "r     192\n",
       "Name: stalk-root, dtype: int64"
      ]
     },
     "execution_count": 4,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df[\"stalk-root\"].value_counts()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "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vPMnuPXWm7u8jquu+gmf+tR2bLbZRuy008f5yU/mTdg7Xmetxx//PzbeeF3+9Kc/zp32i19cw6677sgHPvBePvrRzTnooH156KEHe/GOSpIkSQs/axYXEU8++Tjnnnsmn/3s/rS2tnLRRRdw2GGf54orrqa1tZVTTz2J2277A/vtdyBvecvbuP76azj88EM455wLWHPNd9dd50knHcesWbM44ohjGDNmDP/3f//h0UenzC3/y1/+zHHHHcWWW36UAw88hAcfvJ8LLzyf559/jsMPP3qedZ1wwjFss8127Lzz7vzsZz/huOOO5sorr2Xppd8IwHXX/ZwzzzydnXbahQ022Ii//vVOzjvvLGbPfoXddtuj4fdh8uS/cvrp32CfffZntdVW56WXXuSee/7Biy/O6PV7KkmSJC3MTBYXEc8++yzf+MY3WX31NQGIWJWddvo4v/zl9bz73evwu9/9mqOPPo6ttvoYABtssBGf+cyn+MEPLuRb3zqv7jr/+c97Oe64k9l44/cDsPba685TfuGF57PWWutw7LEnALDhhu8B4IILvs1nPrP33EQQYMcdd+ZjH9u2iu0dbLPNFtx++x/5+Md3oK2tjYsv/h4f+cjWfP7zXwBg/fU3ZMaMGVx66X+x446fprW1sWHn77vvXt7+9pXZbbc9507beONNGlpWkiRJWpTYDHURMX78hLmJIsAyy7yJVVZ5B/fddy///Oe9tLe3s9lmH5xb3tLSwmabfZC7757c7TpXWmkVLrjgPG688XqeeOKJecpeffVV/v3vf82zToDNN/8wbW1t3HPPP+aZvv76G879f+zYcYwbN56nnnoKgKeeeoqnn57KZptt3mldH+LFF1/koYceaPBdgJVXXoX770/OOeebTJ78V1555ZWGl5UkSZIWJdYsLiLGjx9fZ9oEpk2bxrRpTzN69OKMGjVqnvIJEyYwc+ZMZs+ezciRI7ss/7WvfYPvfe87nHPOt5gx4wVWWmkVDjroUNZdd32ee+5Z5syZw4QJE7psE+D555+bZ/qYMUvO83yxxRZj9uxyr61p056ulp3YKb6J1bqen+/r77Deehtw9NHH8dOf/pif/vTHjB69OB/+8FZ87nOHMHr06IbXI2nBNGtUaUmS1DiTxUXE9OnT60x7hhVWWJGJE5fi5ZdfYubMmfMkjM888wyjRo2qmygCTJq0NMccczxtbW3cd9+9XHzx9/jyl7/IVVf9grFjxzFixIgu250+/RkA3vCGsQ3HPnHiUvMs+1p806p1vQGAkSNHMmfOvDWF9RLJrbb6GFtt9TGmT5/OrbfezLnnfosllliCAw74fMMxSVowzRpVeii6/LRdmh2CJEl12Qx1ETF9+jP84x9/n/v8iSee4N///hfvfOdqrLrqagwbNozf//53c8vb29u55ZabWGON+oPb1GppaeFd71qdvfb6LDNnzuSJJ55g+PDhRKw6zzoBbr75t3Pnb9TSSy/NUktNqrOu37HEEkuw4oorASV5ffzxx5k1a9bcee64Y94RXWuNHz+ej3/8E6y55lpMmfJQt/NJkiRJiyJrFhcR48aN48QTv8o++xxQjYZ6PuPHT2CrrbamtbWVD35wC84883ReeunFajTUn/PII1M47LCj6q5vxowZfPGLB7Hllh/lbW9blldeeYUf//hSJk6cyPLLLw/A3nvvxxe/eBBf//oJbL75h3nwwQe48MLz2Xrrj88zuM38tLS0sNde+3L66V9n7NhxrLfeBvztb3dxzTU/Y999Pzd3cJv3v39TLrroAk499SS22upj3H9/cuON18+zrosuuoDnn3+OtdZah7Fjx3H//cnkyX9l//0P6tsbK0mSJC2kTBb7YOasV5rSbGjmrL4PxvLGN76J3Xffk+9+9zyefPJx3vGOVTn++JPnJlpHHnks3/3uOVxyyUXMmPECK664Eqeddla3t80YOXIkb3/7Svz0p1fw1FNPMmrUKFZbbXW+9a1v09pamrKuv/6GnHDC1/nBDy7iN7/5JePHT2CnnXZh773363X822yzHa+8Mpsrr7yCn/70CiZNeiMHHXQoO+302uew4oorcdRRX+WSSy7k1ltvZp111uOoo77KAQfsPXeed7zjnVx55eXcdNNveOmll3jjG5dhzz335ZOf/HSvY5IkSZIWZsPa29ubHUMzLQ88PG3aDNraur4PTzzxCMsss9ygB9XfTj75eB566EEuuuhHzQ7ldWth2RekoWLSpCXts1i5/LRduOu0fZodxpCwzhEXMnXqC80OQ5IWKS0tw5g4cQzACsCUecqaEZAkSZIkaWgzWZQkSZIkdWGfxUXAMccc3+wQJEmSJL3OWLMoSZIkSerCZFGSJEmS1MWgN0ONiDOAT1BGIl09M++JiOWBa2pm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\n",
      "text/plain": [
       "<Figure size 1080x360 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "plt.figure(figsize=(15,5));\n",
    "sns.countplot(df[\"stalk-root\"], hue=df[\"class\"]);\n",
    "plt.title(\"number of different stalk-roots dependent of class\", size=\"20\");\n",
    "plt.xlabel(\"Stalk-root shape\", size=\"15\");\n",
    "plt.ylabel(\"Number of instances\", size=\"15\"); \n",
    "plt.legend([\"poisonous\", \"edible\"], prop={'size': 15});\n",
    "plt.xticks(size=\"12\");\n"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "It is important to look into whether or not the dataset is imbalanced. If the data is imbalanced, one class has a lot more instances than the other. The best classifcation situation would be one where there are about equally as many instances of both classes.\n",
    "\n",
    "The plot below shows the distribution of the two different classes we are going to predict. There seem to be slightly more edible mushrooms than poisonous ones. It seems like we have a fairly balanced dataset."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 1080x360 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "plt.figure(figsize=(15,5));\n",
    "sns.countplot(df[\"class\"]);\n",
    "plt.title(\"Class distribution\", size=\"20\");\n",
    "plt.xlabel(\"Class, poisonous, edible\", size=\"15\");\n",
    "plt.ylabel(\"Number of instances\", size=\"15\"); \n",
    "plt.xticks(size=\"12\");"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 1080x360 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "plt.figure(figsize=(15,5));\n",
    "sns.countplot(df[\"odor\"], hue=df[\"class\"]);\n",
    "plt.title(\"Class distribution\", size=\"20\");\n",
    "plt.xlabel(\"Class, poisonous, edible\", size=\"15\");\n",
    "plt.ylabel(\"Number of instances\", size=\"15\"); \n",
    "plt.xticks(size=\"12\");"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "When looking at the subplots below, we can tell that some columns are very skewed towards one value. \n",
    "- veil-type only contains values of type p\n",
    "- veil-color contains an overwhelming amount of white instances\n",
    "- gill-attachment contains a very skewed amount of values for free\n",
    "- ring-number is very skewed towards one ring"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "p    8124\n",
       "Name: veil-type, dtype: int64"
      ]
     },
     "execution_count": 8,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df[\"veil-type\"].value_counts()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "w    7924\n",
       "n      96\n",
       "o      96\n",
       "y       8\n",
       "Name: veil-color, dtype: int64"
      ]
     },
     "execution_count": 9,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df[\"veil-color\"].value_counts()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "f    7914\n",
       "a     210\n",
       "Name: gill-attachment, dtype: int64"
      ]
     },
     "execution_count": 10,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df[\"gill-attachment\"].value_counts()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "o    7488\n",
       "t     600\n",
       "n      36\n",
       "Name: ring-number, dtype: int64"
      ]
     },
     "execution_count": 11,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df[\"ring-number\"].value_counts()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 12,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 1080x1800 with 22 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "fig, ax = plt.subplots(11,2, figsize=(15,25))\n",
    "fig.suptitle(\"Count plot of all features\", size=25, y=0.92)\n",
    "\n",
    "sns.countplot(df[\"class\"], ax=ax[0,0])\n",
    "ax[0,0].set_title(\"count of edible/poisonous\", size=\"15\");\n",
    "ax[0,0].set_xlabel(\"class\", size=\"12\")\n",
    "\n",
    "sns.countplot(df[\"cap-shape\"], ax=ax[0,1])\n",
    "ax[0,1].set_title(\"count of cap-shapes\", size=\"15\");\n",
    "ax[0,1].set_xlabel(\"cap-shape\", size=\"12\")\n",
    "\n",
    "sns.countplot(df[\"cap-surface\"], ax=ax[1,0])\n",
    "ax[1,0].set_title(\"count of cap-surface\", size=\"15\");\n",
    "ax[1,0].set_xlabel(\"cap-surface\", size=\"12\")\n",
    "\n",
    "sns.countplot(df[\"cap-color\"], ax=ax[1,1])\n",
    "ax[1,1].set_title(\"count of cap-color\", size=\"15\");\n",
    "ax[1,1].set_xlabel(\"cap-color\", size=\"12\")\n",
    "\n",
    "sns.countplot(df[\"bruises\"], ax=ax[2,0])\n",
    "ax[2,0].set_title(\"count of bruising\", size=\"15\");\n",
    "ax[2,0].set_xlabel(\"bruises\", size=\"12\")\n",
    "\n",
    "sns.countplot(df[\"odor\"], ax=ax[2,1])\n",
    "ax[2,1].set_title(\"count of odor\", size=\"15\");\n",
    "ax[2,1].set_xlabel(\"odors\", size=\"12\")\n",
    "\n",
    "sns.countplot(df[\"gill-attachment\"], ax=ax[3,0])\n",
    "ax[3,0].set_title(\"count of gill-attachment\", size=\"15\");\n",
    "ax[3,0].set_xlabel(\"gill-attachment\", size=\"12\")\n",
    "\n",
    "sns.countplot(df[\"gill-spacing\"], ax=ax[3,1])\n",
    "ax[3,1].set_title(\"count of gill-spacing\", size=\"15\");\n",
    "ax[3,1].set_xlabel(\"gill-spacing\", size=\"12\")\n",
    "\n",
    "sns.countplot(df[\"gill-size\"], ax=ax[4,0])\n",
    "ax[4,0].set_title(\"count of gill-size\", size=\"15\");\n",
    "ax[4,0].set_xlabel(\"gill-size\", size=\"14\")\n",
    "\n",
    "sns.countplot(df[\"gill-color\"], ax=ax[4,1])\n",
    "ax[4,1].set_title(\"count of gill-color\", size=\"15\");\n",
    "ax[4,1].set_xlabel(\"gill-colors\", size=\"12\")\n",
    "\n",
    "sns.countplot(df[\"stalk-shape\"], ax=ax[5,0])\n",
    "ax[5,0].set_title(\"count of stalk-shape\", size=\"15\");\n",
    "ax[5,0].set_xlabel(\"stalk-shapes\", size=\"12\")\n",
    "\n",
    "sns.countplot(df[\"stalk-root\"], ax=ax[5,1])\n",
    "ax[5,1].set_title(\"count of stalk-root\", size=\"15\");\n",
    "ax[5,1].set_xlabel(\"stalk-root\", size=\"12\")\n",
    "\n",
    "sns.countplot(df[\"stalk-surface-above-ring\"], ax=ax[6,0])\n",
    "ax[6,0].set_title(\"count of stalk-surface-above-ring\", size=\"15\");\n",
    "ax[6,0].set_xlabel(\"stalk-surface-above-ring\", size=\"12\")\n",
    "\n",
    "sns.countplot(df[\"stalk-surface-below-ring\"], ax=ax[6,1])\n",
    "ax[6,1].set_title(\"count of stalk-surface-below-ring\", size=\"15\");\n",
    "ax[6,1].set_xlabel(\"stalk-surface-below-ring\", size=\"12\")\n",
    "\n",
    "sns.countplot(df[\"stalk-color-above-ring\"], ax=ax[7,0])\n",
    "ax[7,0].set_title(\"count of stalk-color-above-ring\", size=\"15\");\n",
    "ax[7,0].set_xlabel(\"stalk-color-above-ring\", size=\"12\")\n",
    "\n",
    "sns.countplot(df[\"stalk-color-below-ring\"], ax=ax[7,1])\n",
    "ax[7,1].set_title(\"count of stalk-color-below-ring\", size=\"15\");\n",
    "ax[7,1].set_xlabel(\"stalk-color-below-ring\", size=\"12\")\n",
    "\n",
    "sns.countplot(df[\"veil-type\"], ax=ax[8,0])\n",
    "ax[8,0].set_title(\"count of veil-type\", size=\"15\");\n",
    "ax[8,0].set_xlabel(\"veil-type\", size=\"12\")\n",
    "\n",
    "sns.countplot(df[\"veil-color\"], ax=ax[8,1])\n",
    "ax[8,1].set_title(\"count of veil-color\", size=\"15\");\n",
    "ax[8,1].set_xlabel(\"veil-color\", size=\"12\");\n",
    "\n",
    "sns.countplot(df[\"ring-number\"], ax=ax[9,0])\n",
    "ax[9,0].set_title(\"count of ring-number\", size=\"15\");\n",
    "ax[9,0].set_xlabel(\"ring-number\", size=\"12\");\n",
    "\n",
    "sns.countplot(df[\"ring-type\"], ax=ax[9,1])\n",
    "ax[9,1].set_title(\"count of ring-type\", size=\"15\");\n",
    "ax[9,1].set_xlabel(\"ring-type\", size=\"12\");\n",
    "\n",
    "sns.countplot(df[\"spore-print-color\"], ax=ax[10,0])\n",
    "ax[10,0].set_title(\"count of spore-print-color\", size=\"15\");\n",
    "ax[10,0].set_xlabel(\"spore-print-color\", size=\"12\");\n",
    "\n",
    "sns.countplot(df[\"population\"], ax=ax[10,1])\n",
    "ax[10,1].set_title(\"count of population\", size=\"15\");\n",
    "ax[10,1].set_xlabel(\"population\", size=\"12\");\n",
    "plt.subplots_adjust(hspace = 0.75)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Data Cleaning"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "- Based on the plots we used to explore the data we can tell that stalk-root had a lot of missing values. it will not be very useful in making predictions. This feature will be dropped.\n",
    "- Veil-type only contains values of type p which means that this column is the same for every instance in the dataset. It is therefore not useful in making predictions and it will be removed. \n",
    "- veil-color had very few values of any other type than white\n",
    "- ring-number had very few values of any other type than one ring\n",
    "- gill-attachment had very few values of any other type than free"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 13,
   "metadata": {},
   "outputs": [],
   "source": [
    "df.drop(columns=[\"veil-type\", \"stalk-root\", \"veil-color\", \"ring-number\", \"gill-attachment\"], inplace=True)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "The target variable is set to 0 = poisonous and 1 = edible."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 14,
   "metadata": {},
   "outputs": [],
   "source": [
    "df[\"class\"] = (df[\"class\"] == \"e\").astype(int)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Since all of the data we are working with is categorical we need to get dummies of the dataframe so we can use it in scikit-learn."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 15,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "(8124, 102)"
      ]
     },
     "execution_count": 15,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "columns = df.columns[df.columns != \"class\"]\n",
    "df1 = df.drop(columns=[\"class\"])\n",
    "df1 = pd.get_dummies(df1, columns=columns)\n",
    "df1.shape"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "The dataframe now has 111 columns/features, that all contain numeric non-categorical data, ready for scikit-learn machine learning."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Model 1 - Naive Bayes"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "For the first model we will be using Naive Bayes, since this is a very fast and easy classification model to implement. However we should be careful of strongly correlated values."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 16,
   "metadata": {},
   "outputs": [],
   "source": [
    "X = df1.values\n",
    "y = df[\"class\"].values\n",
    "X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "#### Accuracy\n",
    "When looking at the baseline accuracy this seems reasonable from the plot we made in exploration of the class variable, where we saw a distribution of almost 50/50. The accuracy of the model however comes out to 0.981 which is very very high. It seems that the data we are working with gives extremely good predictions when using all the features of the set."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 17,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "The baseline accuracy is: 0.481\n",
      "The accuracy of the model is: 0.981\n"
     ]
    }
   ],
   "source": [
    "clf = GaussianNB()\n",
    "clf.fit(X_train, y_train)\n",
    "y_pred = clf.predict(X_test)\n",
    "\n",
    "print(\"The baseline accuracy is: {:.3}\".format(1 - y_train.mean()))\n",
    "print(\"The accuracy of the model is: {:.3}\".format(clf.score(X_test, y_test)))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 18,
   "metadata": {},
   "outputs": [],
   "source": [
    "def print_conf_mtx(y_true, y_pred, classes=None):\n",
    "    \"\"\" Print a confusion matrix (two classes only). \"\"\"\n",
    "    \n",
    "    if not classes:\n",
    "        classes = ['poisonous', ' edible']\n",
    "   \t \n",
    "    # formatting\n",
    "    max_class_len = max([len(s) for s in classes])\n",
    "    m = max(max_class_len, len('predicted')//2 + 1)\n",
    "    n = max(len('actual')+1, max_class_len)\n",
    "    left   \t= '{:<10s}'.replace('10',str(n))\n",
    "    right  \t= '{:>10s}'.replace('10',str(m))\n",
    "    big_center = '{:^20s}'.replace('20',str(m*2))\n",
    "    \n",
    "    cm = confusion_matrix(y_test, y_pred)\n",
    "    print((left+big_center).format('', 'predicted'))\n",
    "    print((left+right+right).format('actual', classes[0], classes[1]))\n",
    "    print((left+right+right).format(classes[0], str(cm[0,0]), str(cm[0,1])))\n",
    "    print((left+right+right).format(classes[1], str(cm[1,0]), str(cm[1,1])))\n",
    "\n",
    "def getPrecision(y_true, y_pred):\n",
    "    cm = confusion_matrix(y_test, y_pred)\n",
    "    pospos = cm[[1],1]\n",
    "    poscol = cm[[0],1]\n",
    "    return (pospos / (pospos + poscol))[0]\n",
    "\n",
    "def getRecall(y_true, y_pred):\n",
    "    cm = confusion_matrix(y_test, y_pred)\n",
    "    #print(cm.shape)\n",
    "    pospos = cm[[1],1]\n",
    "    posrow = cm[[1],0]\n",
    "    return (pospos / (pospos + posrow))[0]\n",
    "\n",
    "def precisionRecallCurve(y_probs, y_test):\n",
    "    thresholds = np.linspace(0,1,100)\n",
    "    precision_vals = []\n",
    "    recall_vals = []\n",
    "    for t in thresholds:\n",
    "        y_pred = (y_probs > t).astype(int)\n",
    "        precision_vals.append(precision_score(y_pred, y_test))\n",
    "        recall_vals.append(recall_score(y_pred, y_test))\n",
    "    ticks = np.linspace(0,1,10)\n",
    "    labels = ticks.round(1)\n",
    "    precision_vals = np.array(precision_vals)\n",
    "    recall_vals = np.array(recall_vals)\n",
    "    precision_vals = np.insert(precision_vals, 0, 1.0)\n",
    "    recall_vals = np.insert(recall_vals, 0, 0)\n",
    "    recall_vals[-1] = 1.0\n",
    "    plt.title(\"Precision/Recall\")\n",
    "    plt.xlabel(\"Recall\")\n",
    "    plt.ylabel(\"Precision\")\n",
    "    plt.fill_between(recall_vals, precision_vals, alpha=0.2, color='b')\n",
    "    plt.plot(recall_vals, precision_vals)\n"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "#### Confusion Matrix\n",
    "\n",
    "When looking at the confusion matrix it looks like both the precision and recall values are very good. The precision is better than the recall, which is very important for this kind of classification problem it is very important to get the ones predicted to be edible to actually be edible because the consequences of false positives in this case could be death."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 19,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "             predicted     \n",
      "actual   poisonous   edible\n",
      "poisonous     1180        1\n",
      " edible         46     1211\n"
     ]
    }
   ],
   "source": [
    "print_conf_mtx(y_test, y_pred)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "#### Precision/Recall\n",
    "\n",
    "- Precision: of the positive predictions, what fraction are correct\n",
    "- Recall: of the positive cases, which fraction are predicted positive\n",
    "\n",
    "We definitely want a higher precision than recall here even though there in this canse only is a very small difference."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 20,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Precision: 1.00\n",
      "Recall: 0.96\n"
     ]
    }
   ],
   "source": [
    "print(\"Precision: {:.2f}\".format(getPrecision(y_test, y_pred)))\n",
    "print(\"Recall: {:.2f}\".format(getRecall(y_test, y_pred)))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 21,
   "metadata": {},
   "outputs": [],
   "source": [
    "te_errs = []\n",
    "tr_errs = []\n",
    "tr_sizes = np.linspace(100, X_train.shape[0], 10).astype(int)\n",
    "for tr_size in tr_sizes:\n",
    "  X_train1 = X_train[:tr_size,:]\n",
    "  y_train1 = y_train[:tr_size]\n",
    "  \n",
    "  clf.fit(X_train1, y_train1)\n",
    "\n",
    "  tr_predicted = clf.predict(X_train1)\n",
    "  err = (tr_predicted != y_train1).mean()\n",
    "  tr_errs.append(err)\n",
    "  \n",
    "  te_predicted = clf.predict(X_test)\n",
    "  err = (te_predicted != y_test).mean()\n",
    "  te_errs.append(err)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "#### Learning Curve\n",
    "\n",
    "Looking at the learning cure we can tell that the error is very small, even at the beginning of the graph. And the graphs are following each other very very closely. "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 22,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 720x360 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "plt.figure(figsize=(10,5))\n",
    "sns.lineplot(x=tr_sizes, y=te_errs);\n",
    "sns.lineplot(x=tr_sizes, y=tr_errs);\n",
    "plt.title(\"Learning curve Model 1\", size=15);\n",
    "plt.xlabel(\"Training set size\", size=15);\n",
    "plt.ylabel(\"Error\", size=15);\n",
    "plt.legend(['Test Data', 'Training Data']);"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Model 2 - Naive Bayes with Feature Selection"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "#### Feature Selection:"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "For the second model we will do feature selection to try to see if we can get the same accuracy as in the previous model with fewer features. When letting the algorithm pick the best 10 columns, we can see that we get an accuracy of 1 at feature number 6."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 23,
   "metadata": {},
   "outputs": [],
   "source": [
    "X = df1.values\n",
    "y = df[\"class\"].values\n",
    "\n",
    "X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)\n",
    "clf2 = GaussianNB()\n",
    "clf2.fit(X_train, y_train);\n",
    "\n",
    "\n",
    "def bestFeatures(num):\n",
    "    remaining = list(range(X_train.shape[1]))\n",
    "    selected = []\n",
    "    n = num\n",
    "    while len(selected) < n:\n",
    "        min_acc = -1e7\n",
    "        for i in remaining:\n",
    "            X_i = X_train[:,selected+[i]]\n",
    "            scores = cross_val_score(GaussianNB(), X_i, y_train,\n",
    "           scoring='accuracy', cv=3)\n",
    "            accuracy = scores.mean() \n",
    "            if accuracy > min_acc:\n",
    "                min_acc = accuracy\n",
    "                i_min = i\n",
    "\n",
    "        remaining.remove(i_min)\n",
    "        selected.append(i_min)\n",
    "        print('num features: {}; accuracy: {:.2f}'.format(len(selected), min_acc))\n",
    "    return selected"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 24,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "num features: 1; accuracy: 0.89\n",
      "num features: 2; accuracy: 0.94\n",
      "num features: 3; accuracy: 0.96\n",
      "num features: 4; accuracy: 0.96\n",
      "num features: 5; accuracy: 0.99\n",
      "num features: 6; accuracy: 1.00\n",
      "num features: 7; accuracy: 1.00\n",
      "num features: 8; accuracy: 1.00\n",
      "num features: 9; accuracy: 1.00\n",
      "num features: 10; accuracy: 1.00\n"
     ]
    }
   ],
   "source": [
    "selected = df1.columns[bestFeatures(10)]"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Here we see that the best features picked by the algorithm are to do with odor, habitat and spore print color. These are features that might be easy for people to identify, however spore prints might not always be present and stalk color below the ring can also be hard to distinguish. We will explore if we can find such features that are easier to observe and give good accuracy in the next model."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 25,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "The best 6 features are: ['odor_n' 'odor_a' 'habitat_m' 'spore-print-color_r' 'odor_l'\n",
      " 'stalk-color-below-ring_y']\n"
     ]
    }
   ],
   "source": [
    "predictors = selected[0:6].values\n",
    "print(\"The best 6 features are:\", predictors)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "#### Correlation Plot of chosen features\n",
    "\n",
    "This heatmap shows the correlation between the features chosen in feature selection. Naive Bayes models don't do well with strongly correlated features, in this plot we can tell that habitat seems to be somewhat correlated with odors almond and anise but the model does not seem to suffer from this."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 26,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 720x360 with 2 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "plt.figure(figsize=(10,5))\n",
    "corr = df1[predictors].corr();\n",
    "sns.heatmap(corr);\n",
    "plt.title(\"Heatmap of features used in model 2\");"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Training our new model using the predictors found by feature selection"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 27,
   "metadata": {},
   "outputs": [],
   "source": [
    "X = df1[predictors].values\n",
    "y = df[\"class\"].values\n",
    "\n",
    "X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.3, random_state=42)\n",
    "\n",
    "clf2 = GaussianNB();\n",
    "clf2.fit(X_train, y_train);"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "#### Accuracy\n",
    "Looking at the accuracy for this model it has gone up by 0.01 percent using these six predictors. This is a very good model for predicting edible mushrooms."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 28,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "The baseline accuracy is: 0.481\n",
      "The accuracy of the model is: 0.996\n"
     ]
    }
   ],
   "source": [
    "y_pred = clf2.predict(X_test)\n",
    "print(\"The baseline accuracy is: {:.3}\".format(1 - y_train.mean()))\n",
    "print(\"The accuracy of the model is: {:.3}\".format(clf2.score(X_test, y_test)))"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "#### Confusion Matrix:\n",
    "\n",
    "Looking at the confusion Matrix for the model we can tell that the results have changed ever so slightly. There is now a better Recall value while Precision has gone a bit down. One could argue that this development is not favourable since it is very bad to predict a mushroom to be edible and it ending up actually being poisonous."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 29,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "             predicted     \n",
      "actual   poisonous   edible\n",
      "poisonous     1172        9\n",
      " edible          0     1257\n"
     ]
    }
   ],
   "source": [
    "print_conf_mtx(y_test, y_pred)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "#### Precision/Recall:\n",
    "\n",
    "The precision recall values are confirming what we saw in the confusion matrix, that Recall is now perfect while Precision has gone down, even if only by a little."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 30,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Precision: 0.993\n",
      "Recall: 1.000\n"
     ]
    }
   ],
   "source": [
    "print(\"Precision: {:.3f}\".format(getPrecision(y_test, y_pred)))\n",
    "print(\"Recall: {:.3f}\".format(getRecall(y_test, y_pred)))"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "#### Learning Curve:\n",
    "\n",
    "The learning curve looks very interesting. We can tell that the error is very very small. We do not seem to have a high variance case since the training error is so low and the gap between the curves is pretty small. It does not look like a high bias problem either since the training error is so low. It looks like the model is fitted very well to the training data and that it works very well for the test data."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 31,
   "metadata": {},
   "outputs": [],
   "source": [
    "te_errs = []\n",
    "tr_errs = []\n",
    "tr_sizes = np.linspace(100, X_train.shape[0], 10).astype(int)\n",
    "for tr_size in tr_sizes:\n",
    "  X_train1 = X_train[:tr_size,:]\n",
    "  y_train1 = y_train[:tr_size]\n",
    "  \n",
    "  clf2.fit(X_train1, y_train1)\n",
    "\n",
    "  tr_predicted = clf2.predict(X_train1)\n",
    "  err = (tr_predicted != y_train1).mean()\n",
    "  tr_errs.append(err)\n",
    "  \n",
    "  te_predicted = clf2.predict(X_test)\n",
    "  err = (te_predicted != y_test).mean()\n",
    "  te_errs.append(err)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 32,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "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evaM6lwlaiIiIlOAkrSpz/NiwOhmzypRExEREclRGqMmIiIiI3bppRfS29tLX18vLS1rWbBgEQDNzcZVV31m1Nf7wx9+x6xZDRxyyKGDHr/uuk/zxBOPU1VVzZ49u6mpqeXMM9/Oqaf+00Gv/dhjj5JOpznuuONHHVeuUKI2Buntbez5w3coff2/ECurijocERGRrLn99rsA2LhxA5dc8m7uvPP747reH/7wvxx55FFDJmoAF1xwMWeeeRYAy5c/z9VXf4Jt27Zx9tnnDnvtxx57lFQqpUSt0Pg9u0htdKTWPUNsycujDkdERCRn3H//z/n5z39CKpWisrKKj3zkEySTTTz55BN8/etfxveD5UYuuuhSpk8v469//QtPPPE49977U84//90H7Slrbj6EK6/8EF/+8uc5++xzaWvbzLXXfopdu3bS09PDK195Mpdd9i+sWOG4//6f4/s+Dz/8V0499TTOOutcPv7xD7Ft2za6u7s57LDD+ehHr6KoKHfTodyNLIfFapJQXEpq00qKlaiJiEgW/eXpjfz5qUG3hRy3Vxw5m5OOmD3m8x9//O/86U+/5+abv0txcTF//vMf+dKXPseNN36bu+++g/PPv4BTTjkN3/fp6uqisrKSE088iSOPPGpvj9lIHHro4bS3b2H79m1UVVVxww3foKysjN7eXv7t3/6ZRx99mOOOO57TTz+DVCrFFVdcCUA6neaaa66nqqqKdDrNZz97Nb/85f28+c1njvk9TzYlamPgxWLE6xeR2rQi6lBERERyxl/+8keWL3dceumFQLB+2K5duwA4+uhjueuu/2TDhvUcd9zxHHro4eO4076Zk6lUmhtv/BrPPvs0AO3tW1ixYvmgjzvT6TR3330njzzyN9LpFNu3b6eysnIccUw+JWpjFJ+1mJ5l9+H37MYrKYs6HBERKRAnHTG+Xq/J5Ps+b3nLW7n44ksPOHb++e/mVa86mb///WG++tUv8fKXv4L3vveyMd3nuef+wcyZdVRVVfPd797G7t27uf3271FSUsL1119LT0/3oOc9+OADPPfcs9x883eYPn06d9xxO5s2tY4phmzR8hxjFG9YAr5PavMLUYciIiKSE17xilfzy1/ez5YtbQCkUimef/45ANauXcPcuUnOPPMszjrrHTz33LMATJ9eTldX14jvsWLFcr71rf/gne8Meu12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\n",
      "text/plain": [
       "<Figure size 720x360 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "plt.figure(figsize=(10,5))\n",
    "sns.lineplot(x=tr_sizes, y=te_errs);\n",
    "sns.lineplot(x=tr_sizes, y=tr_errs);\n",
    "plt.title(\"Learning curve Model 2\", size=15);\n",
    "plt.xlabel(\"Training set size\", size=15);\n",
    "plt.ylabel(\"Error\", size=15);\n",
    "plt.legend(['Test Data', 'Training Data']);"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Model 3 - Naive Bayes - Making the most usable prediction model"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "In this model we will try to make to most usable prediction model possible by selecting features that are easy for people going shrooming to distiguish between. The features we selected are: \n",
    "- cap-surface\n",
    "- odor \n",
    "- cap-shape\n",
    "These are all features that should be easily distinguishable when going out looking for mushrooms and it should be easy for people to take note of these characteristics. We are getting dummy variables for these featues as they are categorical in nature."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 33,
   "metadata": {},
   "outputs": [],
   "source": [
    "cols = [\"cap-surface\", \"odor\", \"cap-shape\"]\n",
    "df3 = df[cols]\n",
    "df3 = pd.get_dummies(df3)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 34,
   "metadata": {},
   "outputs": [],
   "source": [
    "X = df3.values\n",
    "\n",
    "X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)\n",
    "clf3 = GaussianNB()\n",
    "clf3.fit(X_train, y_train);"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "#### Forward feature selection: \n",
    "\n",
    "When doing forward feature selection based on the features we chose for this model, we get an accuracy of 99 when we get to 5 features."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 35,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "num features: 1; accuracy: 0.89\n",
      "num features: 2; accuracy: 0.94\n",
      "num features: 3; accuracy: 0.96\n",
      "num features: 4; accuracy: 0.96\n",
      "num features: 5; accuracy: 0.99\n"
     ]
    }
   ],
   "source": [
    "selected = df3.columns[bestFeatures(5)]"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "The best five features to use for these predictions according to forward feature selection are: \n",
    "- odor none\n",
    "- odor almond\n",
    "- cap shape bell\n",
    "- cap shape conical\n",
    "- odor anise"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 36,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "The best 5 features are: ['odor_n' 'odor_a' 'cap-shape_b' 'cap-shape_c' 'odor_l']\n"
     ]
    }
   ],
   "source": [
    "predictors3 = selected[0:10].values\n",
    "print(\"The best 5 features are:\", predictors3)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "#### Correlation Plot\n",
    "\n",
    "It does not seem like any of the features used in this model are strongly correlated. This is very good for Naive bayes predictions."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 37,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 720x360 with 2 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "plt.figure(figsize=(10,5))\n",
    "corr3 = df3[predictors3].corr();\n",
    "sns.heatmap(corr3);\n",
    "plt.title(\"Heatmap of features used in model2\");"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "#### Accuracy\n",
    "\n",
    "The accuracy of the model is high at 0.985. This isn't a perfect model like we almost acheived with forward feature selection in the previous model, but it is still high enough to be reasonably good at predicting edible mushrooms."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 38,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "The baseline accuracy is: 0.481\n",
      "The accuracy of the model is: 0.985\n"
     ]
    }
   ],
   "source": [
    "y_pred = clf3.predict(X_test)\n",
    "print(\"The baseline accuracy is: {:.3}\".format(1 - y_train.mean()))\n",
    "print(\"The accuracy of the model is: {:.3}\".format(clf3.score(X_test, y_test)))"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "#### Confusion Matrix\n",
    "\n",
    "When looking at the confusion matrix produced for this model we can tell that the recall is perfect while the precision is a little bit off. It is not much but there are still 36 mushrooms that will be classified aas edible that are actually poisonous. This is not a good outcome. However it is a very very small fraction. One could argue that is would be very important to get precision to 1 here, because you would not want people believing in your model and then ending up being severely poisoned. "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 39,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "             predicted     \n",
      "actual   poisonous   edible\n",
      "poisonous     1145       36\n",
      " edible          0     1257\n"
     ]
    }
   ],
   "source": [
    "print_conf_mtx(y_test, y_pred)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "#### Precision/ Recall\n",
    "\n",
    "Like stated in the previous paragraph we can tell that Recall is perfect and we have a little bit to go on when it comes to precision. Hwever, precision for this classification is still really good at 0.97."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 40,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Precision: 0.972\n",
      "Recall: 1.000\n"
     ]
    }
   ],
   "source": [
    "print(\"Precision: {:.3f}\".format(getPrecision(y_test, y_pred)))\n",
    "print(\"Recall: {:.3f}\".format(getRecall(y_test, y_pred)))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 41,
   "metadata": {},
   "outputs": [],
   "source": [
    "te_errs = []\n",
    "tr_errs = []\n",
    "tr_sizes = np.linspace(100, X_train.shape[0], 10).astype(int)\n",
    "for tr_size in tr_sizes:\n",
    "  X_train1 = X_train[:tr_size,:]\n",
    "  y_train1 = y_train[:tr_size]\n",
    "  \n",
    "  clf3.fit(X_train1, y_train1)\n",
    "\n",
    "  tr_predicted = clf3.predict(X_train1)\n",
    "  err = (tr_predicted != y_train1).mean()\n",
    "  tr_errs.append(err)\n",
    "  \n",
    "  te_predicted = clf3.predict(X_test)\n",
    "  err = (te_predicted != y_test).mean()\n",
    "  te_errs.append(err)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "#### Learning Curve\n",
    "\n",
    "This learning curve also shows us a very small error. However, it does look like we get a very low error for the training data that actaully goes back up a bit when getting to around 1500 in training size. We also have a point where the two curves intersect at right before 1000 instances. The two lines do however smooth out and stay fairly close to each other as more data is provided. It looks like it is stabilizing at around an error of 0.0150."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 42,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 720x360 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "plt.figure(figsize=(10,5))\n",
    "sns.lineplot(x=tr_sizes, y=te_errs);\n",
    "sns.lineplot(x=tr_sizes, y=tr_errs);\n",
    "plt.title(\"Learning curve Model 1\", size=15);\n",
    "plt.xlabel(\"Training set size\", size=15);\n",
    "plt.ylabel(\"Error\", size=15);\n",
    "plt.legend(['Test Data', 'Training Data']);"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "#### Conclusion Model 3: \n",
    "\n",
    "It looks like overall we ended up with a marginally lower accuracy and a little bit of a lower precision for this model than for the last model using feature selection on all features in the dataset. However this might be a more usable model for poeple actually in the woods looking for mushrooms. The features we worked with in this case are easier to distinguish than the ones found in the previous even more accurate model."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Tree Classifiers"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Model 4 - Tree Classification - All Features"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "For the fourth model we are using a tree classifier to compare to our earlier naive bayes models. First we are making a model using all the features in the dataset to see if we get as good of an accuracy here as we did with Naive Bayes."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 43,
   "metadata": {},
   "outputs": [],
   "source": [
    "np.random.seed(42)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 44,
   "metadata": {},
   "outputs": [],
   "source": [
    "X = df1.values\n",
    "\n",
    "X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.3, random_state=42)\n",
    "\n",
    "clf4 = DecisionTreeClassifier(max_depth=3, random_state=0);\n",
    "clf4.fit(X_train, y_train);"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "#### Tree Graph\n",
    "\n",
    "A graph displaying our model fit to the training data is produced. From the graph we can tell that our baseline accuracy should be 0.48 since looking at all data points, 48% of the mushrooms are poisonous. So if we haven't looked at the dataset at all before guessing that a mushroom is poisonous, we have a 48% chance of being right."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 45,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/svg+xml": [
       "<?xml version=\"1.0\" encoding=\"UTF-8\" standalone=\"no\"?>\n",
       "<!DOCTYPE svg PUBLIC \"-//W3C//DTD SVG 1.1//EN\"\n",
       " \"http://www.w3.org/Graphics/SVG/1.1/DTD/svg11.dtd\">\n",
       "<!-- Generated by graphviz version 2.40.1 (20161225.0304)\n",
       " -->\n",
       "<!-- Title: Tree Pages: 1 -->\n",
       "<svg width=\"706pt\" height=\"433pt\"\n",
       " viewBox=\"0.00 0.00 706.00 433.00\" xmlns=\"http://www.w3.org/2000/svg\" xmlns:xlink=\"http://www.w3.org/1999/xlink\">\n",
       "<g id=\"graph0\" class=\"graph\" transform=\"scale(1 1) rotate(0) translate(4 429)\">\n",
       "<title>Tree</title>\n",
       "<polygon fill=\"#ffffff\" stroke=\"transparent\" points=\"-4,4 -4,-429 702,-429 702,4 -4,4\"/>\n",
       "<!-- 0 -->\n",
       "<g id=\"node1\" class=\"node\">\n",
       "<title>0</title>\n",
       "<path fill=\"#f1f8fd\" stroke=\"#000000\" d=\"M410,-425C410,-425 302,-425 302,-425 296,-425 290,-419 290,-413 290,-413 290,-354 290,-354 290,-348 296,-342 302,-342 302,-342 410,-342 410,-342 416,-342 422,-348 422,-354 422,-354 422,-413 422,-413 422,-419 416,-425 410,-425\"/>\n",
       "<text text-anchor=\"start\" x=\"317.5\" y=\"-409.8\" font-family=\"Helvetica,sans-Serif\" font-size=\"14.00\" fill=\"#000000\">odor_n ≤ 0.5</text>\n",
       "<text text-anchor=\"start\" x=\"328\" y=\"-394.8\" font-family=\"Helvetica,sans-Serif\" font-size=\"14.00\" fill=\"#000000\">gini = 0.5</text>\n",
       "<text text-anchor=\"start\" x=\"299\" y=\"-379.8\" font-family=\"Helvetica,sans-Serif\" font-size=\"14.00\" fill=\"#000000\">samples = 100.0%</text>\n",
       "<text text-anchor=\"start\" x=\"298\" y=\"-364.8\" font-family=\"Helvetica,sans-Serif\" font-size=\"14.00\" fill=\"#000000\">value = [0.48, 0.52]</text>\n",
       "<text text-anchor=\"start\" x=\"313.5\" y=\"-349.8\" font-family=\"Helvetica,sans-Serif\" font-size=\"14.00\" fill=\"#000000\">class = Edible</text>\n",
       "</g>\n",
       "<!-- 1 -->\n",
       "<g id=\"node2\" class=\"node\">\n",
       "<title>1</title>\n",
       "<path fill=\"#ea9a61\" stroke=\"#000000\" d=\"M317,-306C317,-306 209,-306 209,-306 203,-306 197,-300 197,-294 197,-294 197,-235 197,-235 197,-229 203,-223 209,-223 209,-223 317,-223 317,-223 323,-223 329,-229 329,-235 329,-235 329,-294 329,-294 329,-300 323,-306 317,-306\"/>\n",
       "<text text-anchor=\"start\" x=\"218\" y=\"-290.8\" font-family=\"Helvetica,sans-Serif\" font-size=\"14.00\" fill=\"#000000\">bruises_f ≤ 0.5</text>\n",
       "<text text-anchor=\"start\" x=\"231\" y=\"-275.8\" font-family=\"Helvetica,sans-Serif\" font-size=\"14.00\" fill=\"#000000\">gini = 0.28</text>\n",
       "<text text-anchor=\"start\" x=\"210\" y=\"-260.8\" font-family=\"Helvetica,sans-Serif\" font-size=\"14.00\" fill=\"#000000\">samples = 56.1%</text>\n",
       "<text text-anchor=\"start\" x=\"205\" y=\"-245.8\" font-family=\"Helvetica,sans-Serif\" font-size=\"14.00\" fill=\"#000000\">value = [0.83, 0.17]</text>\n",
       "<text text-anchor=\"start\" x=\"208\" y=\"-230.8\" font-family=\"Helvetica,sans-Serif\" font-size=\"14.00\" fill=\"#000000\">class = Poisonous</text>\n",
       "</g>\n",
       "<!-- 0&#45;&gt;1 -->\n",
       "<g id=\"edge1\" class=\"edge\">\n",
       "<title>0&#45;&gt;1</title>\n",
       "<path fill=\"none\" stroke=\"#000000\" d=\"M323.4731,-341.8796C316.5049,-332.9633 309.0753,-323.4565 301.8944,-314.268\"/>\n",
       "<polygon fill=\"#000000\" stroke=\"#000000\" points=\"304.5812,-312.0221 295.6657,-306.2981 299.0657,-316.3326 304.5812,-312.0221\"/>\n",
       "<text text-anchor=\"middle\" x=\"293.2311\" y=\"-322.4493\" font-family=\"Helvetica,sans-Serif\" font-size=\"14.00\" fill=\"#000000\">True</text>\n",
       "</g>\n",
       "<!-- 6 -->\n",
       "<g id=\"node7\" class=\"node\">\n",
       "<title>6</title>\n",
       "<path fill=\"#40a0e6\" stroke=\"#000000\" d=\"M517.5,-306C517.5,-306 380.5,-306 380.5,-306 374.5,-306 368.5,-300 368.5,-294 368.5,-294 368.5,-235 368.5,-235 368.5,-229 374.5,-223 380.5,-223 380.5,-223 517.5,-223 517.5,-223 523.5,-223 529.5,-229 529.5,-235 529.5,-235 529.5,-294 529.5,-294 529.5,-300 523.5,-306 517.5,-306\"/>\n",
       "<text text-anchor=\"start\" x=\"376.5\" y=\"-290.8\" font-family=\"Helvetica,sans-Serif\" font-size=\"14.00\" fill=\"#000000\">spore&#45;print&#45;color_r ≤ 0.5</text>\n",
       "<text text-anchor=\"start\" x=\"417\" y=\"-275.8\" font-family=\"Helvetica,sans-Serif\" font-size=\"14.00\" fill=\"#000000\">gini = 0.06</text>\n",
       "<text text-anchor=\"start\" x=\"396\" y=\"-260.8\" font-family=\"Helvetica,sans-Serif\" font-size=\"14.00\" fill=\"#000000\">samples = 43.9%</text>\n",
       "<text text-anchor=\"start\" x=\"391\" y=\"-245.8\" font-family=\"Helvetica,sans-Serif\" font-size=\"14.00\" fill=\"#000000\">value = [0.03, 0.97]</text>\n",
       "<text text-anchor=\"start\" x=\"406.5\" y=\"-230.8\" font-family=\"Helvetica,sans-Serif\" font-size=\"14.00\" fill=\"#000000\">class = Edible</text>\n",
       "</g>\n",
       "<!-- 0&#45;&gt;6 -->\n",
       "<g id=\"edge6\" class=\"edge\">\n",
       "<title>0&#45;&gt;6</title>\n",
       "<path fill=\"none\" stroke=\"#000000\" d=\"M388.5269,-341.8796C395.4951,-332.9633 402.9247,-323.4565 410.1056,-314.268\"/>\n",
       "<polygon fill=\"#000000\" stroke=\"#000000\" points=\"412.9343,-316.3326 416.3343,-306.2981 407.4188,-312.0221 412.9343,-316.3326\"/>\n",
       "<text text-anchor=\"middle\" x=\"418.7689\" y=\"-322.4493\" font-family=\"Helvetica,sans-Serif\" font-size=\"14.00\" fill=\"#000000\">False</text>\n",
       "</g>\n",
       "<!-- 2 -->\n",
       "<g id=\"node3\" class=\"node\">\n",
       "<title>2</title>\n",
       "<path fill=\"#c1e1f7\" stroke=\"#000000\" d=\"M170,-187C170,-187 62,-187 62,-187 56,-187 50,-181 50,-175 50,-175 50,-116 50,-116 50,-110 56,-104 62,-104 62,-104 170,-104 170,-104 176,-104 182,-110 182,-116 182,-116 182,-175 182,-175 182,-181 176,-187 170,-187\"/>\n",
       "<text text-anchor=\"start\" x=\"79.5\" y=\"-171.8\" font-family=\"Helvetica,sans-Serif\" font-size=\"14.00\" fill=\"#000000\">odor_f ≤ 0.5</text>\n",
       "<text text-anchor=\"start\" x=\"84\" y=\"-156.8\" font-family=\"Helvetica,sans-Serif\" font-size=\"14.00\" fill=\"#000000\">gini = 0.48</text>\n",
       "<text text-anchor=\"start\" x=\"63\" y=\"-141.8\" font-family=\"Helvetica,sans-Serif\" font-size=\"14.00\" fill=\"#000000\">samples = 16.0%</text>\n",
       "<text text-anchor=\"start\" x=\"58\" y=\"-126.8\" font-family=\"Helvetica,sans-Serif\" font-size=\"14.00\" fill=\"#000000\">value = [0.41, 0.59]</text>\n",
       "<text text-anchor=\"start\" x=\"73.5\" y=\"-111.8\" font-family=\"Helvetica,sans-Serif\" font-size=\"14.00\" fill=\"#000000\">class = Edible</text>\n",
       "</g>\n",
       "<!-- 1&#45;&gt;2 -->\n",
       "<g id=\"edge2\" class=\"edge\">\n",
       "<title>1&#45;&gt;2</title>\n",
       "<path fill=\"none\" stroke=\"#000000\" d=\"M211.5865,-222.8796C199.9048,-213.4229 187.4025,-203.302 175.419,-193.6011\"/>\n",
       "<polygon fill=\"#000000\" stroke=\"#000000\" points=\"177.6076,-190.8697 167.6329,-187.2981 173.2032,-196.3105 177.6076,-190.8697\"/>\n",
       "</g>\n",
       "<!-- 5 -->\n",
       "<g id=\"node6\" class=\"node\">\n",
       "<title>5</title>\n",
       "<path fill=\"#e58139\" stroke=\"#000000\" d=\"M314,-179.5C314,-179.5 212,-179.5 212,-179.5 206,-179.5 200,-173.5 200,-167.5 200,-167.5 200,-123.5 200,-123.5 200,-117.5 206,-111.5 212,-111.5 212,-111.5 314,-111.5 314,-111.5 320,-111.5 326,-117.5 326,-123.5 326,-123.5 326,-167.5 326,-167.5 326,-173.5 320,-179.5 314,-179.5\"/>\n",
       "<text text-anchor=\"start\" x=\"235\" y=\"-164.3\" font-family=\"Helvetica,sans-Serif\" font-size=\"14.00\" fill=\"#000000\">gini = 0.0</text>\n",
       "<text text-anchor=\"start\" x=\"210\" y=\"-149.3\" font-family=\"Helvetica,sans-Serif\" font-size=\"14.00\" fill=\"#000000\">samples = 40.2%</text>\n",
       "<text text-anchor=\"start\" x=\"212.5\" y=\"-134.3\" font-family=\"Helvetica,sans-Serif\" font-size=\"14.00\" fill=\"#000000\">value = [1.0, 0.0]</text>\n",
       "<text text-anchor=\"start\" x=\"208\" y=\"-119.3\" font-family=\"Helvetica,sans-Serif\" font-size=\"14.00\" fill=\"#000000\">class = Poisonous</text>\n",
       "</g>\n",
       "<!-- 1&#45;&gt;5 -->\n",
       "<g id=\"edge5\" class=\"edge\">\n",
       "<title>1&#45;&gt;5</title>\n",
       "<path fill=\"none\" stroke=\"#000000\" d=\"M263,-222.8796C263,-212.2134 263,-200.7021 263,-189.9015\"/>\n",
       "<polygon fill=\"#000000\" stroke=\"#000000\" points=\"266.5001,-189.8149 263,-179.8149 259.5001,-189.815 266.5001,-189.8149\"/>\n",
       "</g>\n",
       "<!-- 3 -->\n",
       "<g id=\"node4\" class=\"node\">\n",
       "<title>3</title>\n",
       "<path fill=\"#77bced\" stroke=\"#000000\" d=\"M120,-68C120,-68 12,-68 12,-68 6,-68 0,-62 0,-56 0,-56 0,-12 0,-12 0,-6 6,0 12,0 12,0 120,0 120,0 126,0 132,-6 132,-12 132,-12 132,-56 132,-56 132,-62 126,-68 120,-68\"/>\n",
       "<text text-anchor=\"start\" x=\"34\" y=\"-52.8\" font-family=\"Helvetica,sans-Serif\" font-size=\"14.00\" fill=\"#000000\">gini = 0.36</text>\n",
       "<text text-anchor=\"start\" x=\"13\" y=\"-37.8\" font-family=\"Helvetica,sans-Serif\" font-size=\"14.00\" fill=\"#000000\">samples = 12.4%</text>\n",
       "<text text-anchor=\"start\" x=\"8\" y=\"-22.8\" font-family=\"Helvetica,sans-Serif\" font-size=\"14.00\" fill=\"#000000\">value = [0.24, 0.76]</text>\n",
       "<text text-anchor=\"start\" x=\"23.5\" y=\"-7.8\" font-family=\"Helvetica,sans-Serif\" font-size=\"14.00\" fill=\"#000000\">class = Edible</text>\n",
       "</g>\n",
       "<!-- 2&#45;&gt;3 -->\n",
       "<g id=\"edge3\" class=\"edge\">\n",
       "<title>2&#45;&gt;3</title>\n",
       "<path fill=\"none\" stroke=\"#000000\" d=\"M97.3818,-103.9815C93.5078,-95.3423 89.4145,-86.2144 85.5094,-77.5059\"/>\n",
       "<polygon fill=\"#000000\" stroke=\"#000000\" points=\"88.6503,-75.9561 81.3649,-68.2637 82.2631,-78.8204 88.6503,-75.9561\"/>\n",
       "</g>\n",
       "<!-- 4 -->\n",
       "<g id=\"node5\" class=\"node\">\n",
       "<title>4</title>\n",
       "<path fill=\"#e58139\" stroke=\"#000000\" d=\"M264,-68C264,-68 162,-68 162,-68 156,-68 150,-62 150,-56 150,-56 150,-12 150,-12 150,-6 156,0 162,0 162,0 264,0 264,0 270,0 276,-6 276,-12 276,-12 276,-56 276,-56 276,-62 270,-68 264,-68\"/>\n",
       "<text text-anchor=\"start\" x=\"185\" y=\"-52.8\" font-family=\"Helvetica,sans-Serif\" font-size=\"14.00\" fill=\"#000000\">gini = 0.0</text>\n",
       "<text text-anchor=\"start\" x=\"163.5\" y=\"-37.8\" font-family=\"Helvetica,sans-Serif\" font-size=\"14.00\" fill=\"#000000\">samples = 3.6%</text>\n",
       "<text text-anchor=\"start\" x=\"162.5\" y=\"-22.8\" font-family=\"Helvetica,sans-Serif\" font-size=\"14.00\" fill=\"#000000\">value = [1.0, 0.0]</text>\n",
       "<text text-anchor=\"start\" x=\"158\" y=\"-7.8\" font-family=\"Helvetica,sans-Serif\" font-size=\"14.00\" fill=\"#000000\">class = Poisonous</text>\n",
       "</g>\n",
       "<!-- 2&#45;&gt;4 -->\n",
       "<g id=\"edge4\" class=\"edge\">\n",
       "<title>2&#45;&gt;4</title>\n",
       "<path fill=\"none\" stroke=\"#000000\" d=\"M152.1192,-103.9815C160.1147,-94.7908 168.5913,-85.0472 176.5979,-75.8436\"/>\n",
       "<polygon fill=\"#000000\" stroke=\"#000000\" points=\"179.2693,-78.1055 183.1921,-68.2637 173.988,-73.5111 179.2693,-78.1055\"/>\n",
       "</g>\n",
       "<!-- 7 -->\n",
       "<g id=\"node8\" class=\"node\">\n",
       "<title>7</title>\n",
       "<path fill=\"#3c9ee5\" stroke=\"#000000\" d=\"M541.5,-187C541.5,-187 356.5,-187 356.5,-187 350.5,-187 344.5,-181 344.5,-175 344.5,-175 344.5,-116 344.5,-116 344.5,-110 350.5,-104 356.5,-104 356.5,-104 541.5,-104 541.5,-104 547.5,-104 553.5,-110 553.5,-116 553.5,-116 553.5,-175 553.5,-175 553.5,-181 547.5,-187 541.5,-187\"/>\n",
       "<text text-anchor=\"start\" x=\"352.5\" y=\"-171.8\" font-family=\"Helvetica,sans-Serif\" font-size=\"14.00\" fill=\"#000000\">stalk&#45;surface&#45;below&#45;ring_y ≤ 0.5</text>\n",
       "<text text-anchor=\"start\" x=\"417\" y=\"-156.8\" font-family=\"Helvetica,sans-Serif\" font-size=\"14.00\" fill=\"#000000\">gini = 0.03</text>\n",
       "<text text-anchor=\"start\" x=\"396\" y=\"-141.8\" font-family=\"Helvetica,sans-Serif\" font-size=\"14.00\" fill=\"#000000\">samples = 43.0%</text>\n",
       "<text text-anchor=\"start\" x=\"391\" y=\"-126.8\" font-family=\"Helvetica,sans-Serif\" font-size=\"14.00\" fill=\"#000000\">value = [0.01, 0.99]</text>\n",
       "<text text-anchor=\"start\" x=\"406.5\" y=\"-111.8\" font-family=\"Helvetica,sans-Serif\" font-size=\"14.00\" fill=\"#000000\">class = Edible</text>\n",
       "</g>\n",
       "<!-- 6&#45;&gt;7 -->\n",
       "<g id=\"edge7\" class=\"edge\">\n",
       "<title>6&#45;&gt;7</title>\n",
       "<path fill=\"none\" stroke=\"#000000\" d=\"M449,-222.8796C449,-214.6838 449,-205.9891 449,-197.5013\"/>\n",
       "<polygon fill=\"#000000\" stroke=\"#000000\" points=\"452.5001,-197.298 449,-187.2981 445.5001,-197.2981 452.5001,-197.298\"/>\n",
       "</g>\n",
       "<!-- 10 -->\n",
       "<g id=\"node11\" class=\"node\">\n",
       "<title>10</title>\n",
       "<path fill=\"#e58139\" stroke=\"#000000\" d=\"M686,-179.5C686,-179.5 584,-179.5 584,-179.5 578,-179.5 572,-173.5 572,-167.5 572,-167.5 572,-123.5 572,-123.5 572,-117.5 578,-111.5 584,-111.5 584,-111.5 686,-111.5 686,-111.5 692,-111.5 698,-117.5 698,-123.5 698,-123.5 698,-167.5 698,-167.5 698,-173.5 692,-179.5 686,-179.5\"/>\n",
       "<text text-anchor=\"start\" x=\"607\" y=\"-164.3\" font-family=\"Helvetica,sans-Serif\" font-size=\"14.00\" fill=\"#000000\">gini = 0.0</text>\n",
       "<text text-anchor=\"start\" x=\"585.5\" y=\"-149.3\" font-family=\"Helvetica,sans-Serif\" font-size=\"14.00\" fill=\"#000000\">samples = 0.9%</text>\n",
       "<text text-anchor=\"start\" x=\"584.5\" y=\"-134.3\" font-family=\"Helvetica,sans-Serif\" font-size=\"14.00\" fill=\"#000000\">value = [1.0, 0.0]</text>\n",
       "<text text-anchor=\"start\" x=\"580\" y=\"-119.3\" font-family=\"Helvetica,sans-Serif\" font-size=\"14.00\" fill=\"#000000\">class = Poisonous</text>\n",
       "</g>\n",
       "<!-- 6&#45;&gt;10 -->\n",
       "<g id=\"edge10\" class=\"edge\">\n",
       "<title>6&#45;&gt;10</title>\n",
       "<path fill=\"none\" stroke=\"#000000\" d=\"M514.0538,-222.8796C533.2224,-210.6158 554.1371,-197.2348 573.1018,-185.1015\"/>\n",
       "<polygon fill=\"#000000\" stroke=\"#000000\" points=\"575.1275,-187.9606 581.6648,-179.623 571.355,-182.0641 575.1275,-187.9606\"/>\n",
       "</g>\n",
       "<!-- 8 -->\n",
       "<g id=\"node9\" class=\"node\">\n",
       "<title>8</title>\n",
       "<path fill=\"#399de5\" stroke=\"#000000\" d=\"M451,-68C451,-68 353,-68 353,-68 347,-68 341,-62 341,-56 341,-56 341,-12 341,-12 341,-6 347,0 353,0 353,0 451,0 451,0 457,0 463,-6 463,-12 463,-12 463,-56 463,-56 463,-62 457,-68 451,-68\"/>\n",
       "<text text-anchor=\"start\" x=\"374\" y=\"-52.8\" font-family=\"Helvetica,sans-Serif\" font-size=\"14.00\" fill=\"#000000\">gini = 0.0</text>\n",
       "<text text-anchor=\"start\" x=\"349\" y=\"-37.8\" font-family=\"Helvetica,sans-Serif\" font-size=\"14.00\" fill=\"#000000\">samples = 42.3%</text>\n",
       "<text text-anchor=\"start\" x=\"351.5\" y=\"-22.8\" font-family=\"Helvetica,sans-Serif\" font-size=\"14.00\" fill=\"#000000\">value = [0.0, 1.0]</text>\n",
       "<text text-anchor=\"start\" x=\"359.5\" y=\"-7.8\" font-family=\"Helvetica,sans-Serif\" font-size=\"14.00\" fill=\"#000000\">class = Edible</text>\n",
       "</g>\n",
       "<!-- 7&#45;&gt;8 -->\n",
       "<g id=\"edge8\" class=\"edge\">\n",
       "<title>7&#45;&gt;8</title>\n",
       "<path fill=\"none\" stroke=\"#000000\" d=\"M431.4989,-103.9815C427.8573,-95.3423 424.0097,-86.2144 420.3388,-77.5059\"/>\n",
       "<polygon fill=\"#000000\" stroke=\"#000000\" points=\"423.5525,-76.119 416.443,-68.2637 417.1021,-78.838 423.5525,-76.119\"/>\n",
       "</g>\n",
       "<!-- 9 -->\n",
       "<g id=\"node10\" class=\"node\">\n",
       "<title>9</title>\n",
       "<path fill=\"#f0b48a\" stroke=\"#000000\" d=\"M601,-68C601,-68 493,-68 493,-68 487,-68 481,-62 481,-56 481,-56 481,-12 481,-12 481,-6 487,0 493,0 493,0 601,0 601,0 607,0 613,-6 613,-12 613,-12 613,-56 613,-56 613,-62 607,-68 601,-68\"/>\n",
       "<text text-anchor=\"start\" x=\"515\" y=\"-52.8\" font-family=\"Helvetica,sans-Serif\" font-size=\"14.00\" fill=\"#000000\">gini = 0.41</text>\n",
       "<text text-anchor=\"start\" x=\"497.5\" y=\"-37.8\" font-family=\"Helvetica,sans-Serif\" font-size=\"14.00\" fill=\"#000000\">samples = 0.7%</text>\n",
       "<text text-anchor=\"start\" x=\"489\" y=\"-22.8\" font-family=\"Helvetica,sans-Serif\" font-size=\"14.00\" fill=\"#000000\">value = [0.71, 0.29]</text>\n",
       "<text text-anchor=\"start\" x=\"492\" y=\"-7.8\" font-family=\"Helvetica,sans-Serif\" font-size=\"14.00\" fill=\"#000000\">class = Poisonous</text>\n",
       "</g>\n",
       "<!-- 7&#45;&gt;9 -->\n",
       "<g id=\"edge9\" class=\"edge\">\n",
       "<title>7&#45;&gt;9</title>\n",
       "<path fill=\"none\" stroke=\"#000000\" d=\"M485.4916,-103.9815C493.5695,-94.7908 502.1334,-85.0472 510.2226,-75.8436\"/>\n",
       "<polygon fill=\"#000000\" stroke=\"#000000\" points=\"512.912,-78.0855 516.8848,-68.2637 507.6542,-73.4642 512.912,-78.0855\"/>\n",
       "</g>\n",
       "</g>\n",
       "</svg>\n"
      ],
      "text/plain": [
       "<graphviz.files.Source at 0x7fb3a778e390>"
      ]
     },
     "execution_count": 45,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "target_names = ['Poisonous', 'Edible']\n",
    "dot_data = export_graphviz(clf4, precision=2,\n",
    "feature_names=df1.columns.values,\n",
    "proportion=True,\n",
    "class_names=target_names,\n",
    "filled=True, rounded=True,\n",
    "special_characters=True)\n",
    "graph = graphviz.Source(dot_data)\n",
    "graph"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "#### Accuracy: \n",
    "\n",
    "The accuracy for this model is the same as we got in the first Naive bayes model. "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 46,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Baseline Accuracy: 0.481\n",
      "Accuracy of Model: 0.981\n"
     ]
    }
   ],
   "source": [
    "print(\"Baseline Accuracy: {:.3}\".format(1-y_train.mean()))\n",
    "y_predict = clf.predict(X_test)\n",
    "print(\"Accuracy of Model: {:.3}\".format((y_predict == y_test).mean()))"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "#### Confusion Matrix:\n",
    "\n",
    "When looking at the confusion matrix it does look identical to the Naive bayes model we made using all features of the dataset. The difference in the two models when it comes to precision and recall seem to be indistinguishable."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 47,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "             predicted     \n",
      "actual   poisonous   edible\n",
      "poisonous     1180        1\n",
      " edible         46     1211\n"
     ]
    }
   ],
   "source": [
    "print_conf_mtx(y_test, y_predict)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "#### Precision/Recall: \n",
    "\n",
    "The precision and recall values are good values for this classification problem as we would want the least amount of people to die, using our classifier. In this case 1 person would eat a mushrom wrongly classified as edible when it is not. This still gives us a precision of 1 when rounded to three decimals."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 48,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Precision: 1.00\n",
      "Recall: 0.96\n"
     ]
    }
   ],
   "source": [
    "print(\"Precision: {:.2f}\".format(getPrecision(y_test, y_predict)))\n",
    "print(\"Recall: {:.2f}\".format(getRecall(y_test, y_predict)))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 49,
   "metadata": {},
   "outputs": [],
   "source": [
    "te_errs = []\n",
    "tr_errs = []\n",
    "tr_sizes = np.linspace(100, X_train.shape[0], 10).astype(int)\n",
    "for tr_size in tr_sizes:\n",
    "  X_train1 = X_train[:tr_size,:]\n",
    "  y_train1 = y_train[:tr_size]\n",
    "  \n",
    "  clf4.fit(X_train1, y_train1)\n",
    "\n",
    "  tr_predicted = clf4.predict(X_train1)\n",
    "  err = (tr_predicted != y_train1).mean()\n",
    "  tr_errs.append(err)\n",
    "  \n",
    "  te_predicted = clf4.predict(X_test)\n",
    "  err = (te_predicted != y_test).mean()\n",
    "  te_errs.append(err)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "#### Learning Curve: \n",
    "\n",
    "The learning curve is what is the most different from our first naive bayes model using all features. We can tell that the error on the training data is pretty much stable except for a slight increase at the end of the graph. But the error is still very very small. The error on the test data decreases rapidly just like in the first model we made but then it straightens out and doesnt have a spike around 2000 instances like our naive bayaes model did. "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 50,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 720x360 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "plt.figure(figsize=(10,5))\n",
    "sns.lineplot(x=tr_sizes, y=te_errs);\n",
    "sns.lineplot(x=tr_sizes, y=tr_errs);\n",
    "plt.title(\"Learning curve Model 4\", size=15);\n",
    "plt.xlabel(\"Training set size\", size=15);\n",
    "plt.ylabel(\"Error\", size=15);\n",
    "plt.legend(['Test Data', 'Training Data']);"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "#### Conclusion Model 5: \n",
    "\n",
    "It seems like the tree classifier does just as well as the bayes model when using all the features of the dataset. The main difference between the two models is the learning cure and how the classifiers develop when getting more data to practice and test on."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Model 5 - Tree Classifier - The Most Usable prediction model"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Earlier we defined a few features that were easy for people to identify when mushroom hunting and that gave a good enough accuracy to where it would be useful. We are now going to take a look at how a tree classifier treats the same features.\n",
    "The features in question are: \n",
    "- cap-surface\n",
    "- odor\n",
    "- cap-shape"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 51,
   "metadata": {},
   "outputs": [],
   "source": [
    "X = df3.values\n",
    "\n",
    "X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 52,
   "metadata": {},
   "outputs": [],
   "source": [
    "def bestTreeFeatures(num):\n",
    "    remaining = list(range(X_train.shape[1]))\n",
    "    selected = []\n",
    "    n = num\n",
    "    while len(selected) < n:\n",
    "        min_acc = -1e7\n",
    "        for i in remaining:\n",
    "            X_i = X_train[:,selected+[i]]\n",
    "            scores = cross_val_score(DecisionTreeClassifier(max_depth=5, random_state=42), X_i, y_train,\n",
    "           scoring='accuracy', cv=3)\n",
    "            accuracy = scores.mean() \n",
    "            if accuracy > min_acc:\n",
    "                min_acc = accuracy\n",
    "                i_min = i\n",
    "\n",
    "        remaining.remove(i_min)\n",
    "        selected.append(i_min)\n",
    "        print('num features: {}; accuracy: {:.2f}'.format(len(selected), min_acc))\n",
    "    return selected"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "#### Forward feature selection\n",
    "\n",
    "When doing forward feature slection using the same 3 features made into dummy variables we can tell that this model actually reaches an accuracy of 0.99 at three features, as opposed to our previous model that needed 5 features. "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 53,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "num features: 1; accuracy: 0.89\n",
      "num features: 2; accuracy: 0.94\n",
      "num features: 3; accuracy: 0.99\n",
      "num features: 4; accuracy: 0.99\n",
      "num features: 5; accuracy: 0.99\n"
     ]
    }
   ],
   "source": [
    "selected = df3.columns[bestTreeFeatures(5)]"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Looking at the three best columns we can tell that this classifier actually gets a 0.99 accuracy only using odor, which is only one feature from our original dataframe. Now made into three columns using dummy variables. This is really interestng. Maybe people can distingush edible mushrooms based only on their smell.\n",
    "- odor none\n",
    "- odor almond\n",
    "- odor anis"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 54,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "The best 3 features are: ['odor_n' 'odor_a' 'odor_l']\n"
     ]
    }
   ],
   "source": [
    "predictors = selected[0:3].values\n",
    "print(\"The best 3 features are:\", predictors)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "#### Tree graph\n",
    "\n",
    "The tree we produce from fitting our model to the trainig data. The max depth is set to 3, and our baseline accuracy is once again 48 percent when predicting that a mushroom is poisonous. At the leaf nodes there is still a good amount of samples left. "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 55,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/svg+xml": [
       "<?xml version=\"1.0\" encoding=\"UTF-8\" standalone=\"no\"?>\n",
       "<!DOCTYPE svg PUBLIC \"-//W3C//DTD SVG 1.1//EN\"\n",
       " \"http://www.w3.org/Graphics/SVG/1.1/DTD/svg11.dtd\">\n",
       "<!-- Generated by graphviz version 2.40.1 (20161225.0304)\n",
       " -->\n",
       "<!-- Title: Tree Pages: 1 -->\n",
       "<svg width=\"428pt\" height=\"433pt\"\n",
       " viewBox=\"0.00 0.00 428.00 433.00\" xmlns=\"http://www.w3.org/2000/svg\" xmlns:xlink=\"http://www.w3.org/1999/xlink\">\n",
       "<g id=\"graph0\" class=\"graph\" transform=\"scale(1 1) rotate(0) translate(4 429)\">\n",
       "<title>Tree</title>\n",
       "<polygon fill=\"#ffffff\" stroke=\"transparent\" points=\"-4,4 -4,-429 424,-429 424,4 -4,4\"/>\n",
       "<!-- 0 -->\n",
       "<g id=\"node1\" class=\"node\">\n",
       "<title>0</title>\n",
       "<path fill=\"#f1f8fd\" stroke=\"#000000\" d=\"M333,-425C333,-425 225,-425 225,-425 219,-425 213,-419 213,-413 213,-413 213,-354 213,-354 213,-348 219,-342 225,-342 225,-342 333,-342 333,-342 339,-342 345,-348 345,-354 345,-354 345,-413 345,-413 345,-419 339,-425 333,-425\"/>\n",
       "<text text-anchor=\"start\" x=\"240.5\" y=\"-409.8\" font-family=\"Helvetica,sans-Serif\" font-size=\"14.00\" fill=\"#000000\">odor_n ≤ 0.5</text>\n",
       "<text text-anchor=\"start\" x=\"251\" y=\"-394.8\" font-family=\"Helvetica,sans-Serif\" font-size=\"14.00\" fill=\"#000000\">gini = 0.5</text>\n",
       "<text text-anchor=\"start\" x=\"222\" y=\"-379.8\" font-family=\"Helvetica,sans-Serif\" font-size=\"14.00\" fill=\"#000000\">samples = 100.0%</text>\n",
       "<text text-anchor=\"start\" x=\"221\" y=\"-364.8\" font-family=\"Helvetica,sans-Serif\" font-size=\"14.00\" fill=\"#000000\">value = [0.48, 0.52]</text>\n",
       "<text text-anchor=\"start\" x=\"236.5\" y=\"-349.8\" font-family=\"Helvetica,sans-Serif\" font-size=\"14.00\" fill=\"#000000\">class = Edible</text>\n",
       "</g>\n",
       "<!-- 1 -->\n",
       "<g id=\"node2\" class=\"node\">\n",
       "<title>1</title>\n",
       "<path fill=\"#ea9a61\" stroke=\"#000000\" d=\"M258,-306C258,-306 150,-306 150,-306 144,-306 138,-300 138,-294 138,-294 138,-235 138,-235 138,-229 144,-223 150,-223 150,-223 258,-223 258,-223 264,-223 270,-229 270,-235 270,-235 270,-294 270,-294 270,-300 264,-306 258,-306\"/>\n",
       "<text text-anchor=\"start\" x=\"165.5\" y=\"-290.8\" font-family=\"Helvetica,sans-Serif\" font-size=\"14.00\" fill=\"#000000\">odor_a ≤ 0.5</text>\n",
       "<text text-anchor=\"start\" x=\"172\" y=\"-275.8\" font-family=\"Helvetica,sans-Serif\" font-size=\"14.00\" fill=\"#000000\">gini = 0.28</text>\n",
       "<text text-anchor=\"start\" x=\"151\" y=\"-260.8\" font-family=\"Helvetica,sans-Serif\" font-size=\"14.00\" fill=\"#000000\">samples = 56.1%</text>\n",
       "<text text-anchor=\"start\" x=\"146\" y=\"-245.8\" font-family=\"Helvetica,sans-Serif\" font-size=\"14.00\" fill=\"#000000\">value = [0.83, 0.17]</text>\n",
       "<text text-anchor=\"start\" x=\"149\" y=\"-230.8\" font-family=\"Helvetica,sans-Serif\" font-size=\"14.00\" fill=\"#000000\">class = Poisonous</text>\n",
       "</g>\n",
       "<!-- 0&#45;&gt;1 -->\n",
       "<g id=\"edge1\" class=\"edge\">\n",
       "<title>0&#45;&gt;1</title>\n",
       "<path fill=\"none\" stroke=\"#000000\" d=\"M252.7686,-341.8796C247.2627,-333.1434 241.3994,-323.8404 235.7176,-314.8253\"/>\n",
       "<polygon fill=\"#000000\" stroke=\"#000000\" points=\"238.6363,-312.8919 230.3433,-306.2981 232.7143,-316.6242 238.6363,-312.8919\"/>\n",
       "<text text-anchor=\"middle\" x=\"225.9196\" y=\"-322.1027\" font-family=\"Helvetica,sans-Serif\" font-size=\"14.00\" fill=\"#000000\">True</text>\n",
       "</g>\n",
       "<!-- 6 -->\n",
       "<g id=\"node7\" class=\"node\">\n",
       "<title>6</title>\n",
       "<path fill=\"#40a0e6\" stroke=\"#000000\" d=\"M408,-298.5C408,-298.5 300,-298.5 300,-298.5 294,-298.5 288,-292.5 288,-286.5 288,-286.5 288,-242.5 288,-242.5 288,-236.5 294,-230.5 300,-230.5 300,-230.5 408,-230.5 408,-230.5 414,-230.5 420,-236.5 420,-242.5 420,-242.5 420,-286.5 420,-286.5 420,-292.5 414,-298.5 408,-298.5\"/>\n",
       "<text text-anchor=\"start\" x=\"322\" y=\"-283.3\" font-family=\"Helvetica,sans-Serif\" font-size=\"14.00\" fill=\"#000000\">gini = 0.06</text>\n",
       "<text text-anchor=\"start\" x=\"301\" y=\"-268.3\" font-family=\"Helvetica,sans-Serif\" font-size=\"14.00\" fill=\"#000000\">samples = 43.9%</text>\n",
       "<text text-anchor=\"start\" x=\"296\" y=\"-253.3\" font-family=\"Helvetica,sans-Serif\" font-size=\"14.00\" fill=\"#000000\">value = [0.03, 0.97]</text>\n",
       "<text text-anchor=\"start\" x=\"311.5\" y=\"-238.3\" font-family=\"Helvetica,sans-Serif\" font-size=\"14.00\" fill=\"#000000\">class = Edible</text>\n",
       "</g>\n",
       "<!-- 0&#45;&gt;6 -->\n",
       "<g id=\"edge6\" class=\"edge\">\n",
       "<title>0&#45;&gt;6</title>\n",
       "<path fill=\"none\" stroke=\"#000000\" d=\"M305.2314,-341.8796C312.231,-330.7735 319.808,-318.7513 326.8556,-307.5691\"/>\n",
       "<polygon fill=\"#000000\" stroke=\"#000000\" points=\"330.002,-309.1411 332.3729,-298.8149 324.08,-305.4087 330.002,-309.1411\"/>\n",
       "<text text-anchor=\"middle\" x=\"336.7967\" y=\"-314.6196\" font-family=\"Helvetica,sans-Serif\" font-size=\"14.00\" fill=\"#000000\">False</text>\n",
       "</g>\n",
       "<!-- 2 -->\n",
       "<g id=\"node3\" class=\"node\">\n",
       "<title>2</title>\n",
       "<path fill=\"#e88d4c\" stroke=\"#000000\" d=\"M187,-187C187,-187 79,-187 79,-187 73,-187 67,-181 67,-175 67,-175 67,-116 67,-116 67,-110 73,-104 79,-104 79,-104 187,-104 187,-104 193,-104 199,-110 199,-116 199,-116 199,-175 199,-175 199,-181 193,-187 187,-187\"/>\n",
       "<text text-anchor=\"start\" x=\"96.5\" y=\"-171.8\" font-family=\"Helvetica,sans-Serif\" font-size=\"14.00\" fill=\"#000000\">odor_l ≤ 0.5</text>\n",
       "<text text-anchor=\"start\" x=\"101\" y=\"-156.8\" font-family=\"Helvetica,sans-Serif\" font-size=\"14.00\" fill=\"#000000\">gini = 0.16</text>\n",
       "<text text-anchor=\"start\" x=\"80\" y=\"-141.8\" font-family=\"Helvetica,sans-Serif\" font-size=\"14.00\" fill=\"#000000\">samples = 51.3%</text>\n",
       "<text text-anchor=\"start\" x=\"75\" y=\"-126.8\" font-family=\"Helvetica,sans-Serif\" font-size=\"14.00\" fill=\"#000000\">value = [0.91, 0.09]</text>\n",
       "<text text-anchor=\"start\" x=\"78\" y=\"-111.8\" font-family=\"Helvetica,sans-Serif\" font-size=\"14.00\" fill=\"#000000\">class = Poisonous</text>\n",
       "</g>\n",
       "<!-- 1&#45;&gt;2 -->\n",
       "<g id=\"edge2\" class=\"edge\">\n",
       "<title>1&#45;&gt;2</title>\n",
       "<path fill=\"none\" stroke=\"#000000\" d=\"M179.1676,-222.8796C174.0091,-214.2335 168.5192,-205.0322 163.1924,-196.1042\"/>\n",
       "<polygon fill=\"#000000\" stroke=\"#000000\" points=\"166.0678,-194.0924 157.9383,-187.2981 160.0564,-197.679 166.0678,-194.0924\"/>\n",
       "</g>\n",
       "<!-- 5 -->\n",
       "<g id=\"node6\" class=\"node\">\n",
       "<title>5</title>\n",
       "<path fill=\"#399de5\" stroke=\"#000000\" d=\"M322.5,-179.5C322.5,-179.5 229.5,-179.5 229.5,-179.5 223.5,-179.5 217.5,-173.5 217.5,-167.5 217.5,-167.5 217.5,-123.5 217.5,-123.5 217.5,-117.5 223.5,-111.5 229.5,-111.5 229.5,-111.5 322.5,-111.5 322.5,-111.5 328.5,-111.5 334.5,-117.5 334.5,-123.5 334.5,-123.5 334.5,-167.5 334.5,-167.5 334.5,-173.5 328.5,-179.5 322.5,-179.5\"/>\n",
       "<text text-anchor=\"start\" x=\"248\" y=\"-164.3\" font-family=\"Helvetica,sans-Serif\" font-size=\"14.00\" fill=\"#000000\">gini = 0.0</text>\n",
       "<text text-anchor=\"start\" x=\"226.5\" y=\"-149.3\" font-family=\"Helvetica,sans-Serif\" font-size=\"14.00\" fill=\"#000000\">samples = 4.9%</text>\n",
       "<text text-anchor=\"start\" x=\"225.5\" y=\"-134.3\" font-family=\"Helvetica,sans-Serif\" font-size=\"14.00\" fill=\"#000000\">value = [0.0, 1.0]</text>\n",
       "<text text-anchor=\"start\" x=\"233.5\" y=\"-119.3\" font-family=\"Helvetica,sans-Serif\" font-size=\"14.00\" fill=\"#000000\">class = Edible</text>\n",
       "</g>\n",
       "<!-- 1&#45;&gt;5 -->\n",
       "<g id=\"edge5\" class=\"edge\">\n",
       "<title>1&#45;&gt;5</title>\n",
       "<path fill=\"none\" stroke=\"#000000\" d=\"M229.1821,-222.8796C235.9017,-211.7735 243.1757,-199.7513 249.9414,-188.5691\"/>\n",
       "<polygon fill=\"#000000\" stroke=\"#000000\" points=\"253.0559,-190.1826 255.238,-179.8149 247.0668,-186.5589 253.0559,-190.1826\"/>\n",
       "</g>\n",
       "<!-- 3 -->\n",
       "<g id=\"node4\" class=\"node\">\n",
       "<title>3</title>\n",
       "<path fill=\"#e58139\" stroke=\"#000000\" d=\"M114,-68C114,-68 12,-68 12,-68 6,-68 0,-62 0,-56 0,-56 0,-12 0,-12 0,-6 6,0 12,0 12,0 114,0 114,0 120,0 126,-6 126,-12 126,-12 126,-56 126,-56 126,-62 120,-68 114,-68\"/>\n",
       "<text text-anchor=\"start\" x=\"35\" y=\"-52.8\" font-family=\"Helvetica,sans-Serif\" font-size=\"14.00\" fill=\"#000000\">gini = 0.0</text>\n",
       "<text text-anchor=\"start\" x=\"10\" y=\"-37.8\" font-family=\"Helvetica,sans-Serif\" font-size=\"14.00\" fill=\"#000000\">samples = 46.7%</text>\n",
       "<text text-anchor=\"start\" x=\"12.5\" y=\"-22.8\" font-family=\"Helvetica,sans-Serif\" font-size=\"14.00\" fill=\"#000000\">value = [1.0, 0.0]</text>\n",
       "<text text-anchor=\"start\" x=\"8\" y=\"-7.8\" font-family=\"Helvetica,sans-Serif\" font-size=\"14.00\" fill=\"#000000\">class = Poisonous</text>\n",
       "</g>\n",
       "<!-- 2&#45;&gt;3 -->\n",
       "<g id=\"edge3\" class=\"edge\">\n",
       "<title>2&#45;&gt;3</title>\n",
       "<path fill=\"none\" stroke=\"#000000\" d=\"M106.9346,-103.9815C101.3955,-95.1585 95.5364,-85.8258 89.9645,-76.9506\"/>\n",
       "<polygon fill=\"#000000\" stroke=\"#000000\" points=\"92.7922,-74.872 84.5108,-68.2637 86.8637,-78.594 92.7922,-74.872\"/>\n",
       "</g>\n",
       "<!-- 4 -->\n",
       "<g id=\"node5\" class=\"node\">\n",
       "<title>4</title>\n",
       "<path fill=\"#399de5\" stroke=\"#000000\" d=\"M249.5,-68C249.5,-68 156.5,-68 156.5,-68 150.5,-68 144.5,-62 144.5,-56 144.5,-56 144.5,-12 144.5,-12 144.5,-6 150.5,0 156.5,0 156.5,0 249.5,0 249.5,0 255.5,0 261.5,-6 261.5,-12 261.5,-12 261.5,-56 261.5,-56 261.5,-62 255.5,-68 249.5,-68\"/>\n",
       "<text text-anchor=\"start\" x=\"175\" y=\"-52.8\" font-family=\"Helvetica,sans-Serif\" font-size=\"14.00\" fill=\"#000000\">gini = 0.0</text>\n",
       "<text text-anchor=\"start\" x=\"153.5\" y=\"-37.8\" font-family=\"Helvetica,sans-Serif\" font-size=\"14.00\" fill=\"#000000\">samples = 4.6%</text>\n",
       "<text text-anchor=\"start\" x=\"152.5\" y=\"-22.8\" font-family=\"Helvetica,sans-Serif\" font-size=\"14.00\" fill=\"#000000\">value = [0.0, 1.0]</text>\n",
       "<text text-anchor=\"start\" x=\"160.5\" y=\"-7.8\" font-family=\"Helvetica,sans-Serif\" font-size=\"14.00\" fill=\"#000000\">class = Edible</text>\n",
       "</g>\n",
       "<!-- 2&#45;&gt;4 -->\n",
       "<g id=\"edge4\" class=\"edge\">\n",
       "<title>2&#45;&gt;4</title>\n",
       "<path fill=\"none\" stroke=\"#000000\" d=\"M159.0654,-103.9815C164.6045,-95.1585 170.4636,-85.8258 176.0355,-76.9506\"/>\n",
       "<polygon fill=\"#000000\" stroke=\"#000000\" points=\"179.1363,-78.594 181.4892,-68.2637 173.2078,-74.872 179.1363,-78.594\"/>\n",
       "</g>\n",
       "</g>\n",
       "</svg>\n"
      ],
      "text/plain": [
       "<graphviz.files.Source at 0x7fb3a82ad5c0>"
      ]
     },
     "execution_count": 55,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "X = df1[predictors].values\n",
    "\n",
    "X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.3, random_state=42)\n",
    "\n",
    "clf5 = DecisionTreeClassifier(max_depth=3, random_state=0);\n",
    "clf5.fit(X_train, y_train);\n",
    "\n",
    "target_names = ['Poisonous', 'Edible']\n",
    "dot_data = export_graphviz(clf5, precision=2,\n",
    "feature_names=df1[predictors].columns.values,\n",
    "proportion=True,\n",
    "class_names=target_names,\n",
    "filled=True, rounded=True,\n",
    "special_characters=True)\n",
    "graph = graphviz.Source(dot_data)\n",
    "graph"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "#### Accuracy \n",
    "\n",
    "The accuracy of this model is 0.001 worse than the previous model we used for predicting on these columns. The accuracy is however still very high."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 56,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Baseline Accuracy: 0.481\n",
      "Model 5 Accuracy: 0.984\n"
     ]
    }
   ],
   "source": [
    "print(\"Baseline Accuracy: {:.3f}\".format(1-y_train.mean()))\n",
    "y_predict5 = clf5.predict(X_test)\n",
    "print(\"Model 5 Accuracy: {:.3f}\".format((y_predict5 == y_test).mean()))"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "#### Confusion Matrix\n",
    "\n",
    "In this confusion matrix we have two more instances that were predicted edible that are in fac poisonous. We can expect precision to be marginally lower in this model than in the Naive Bayes model. Recall is still at 1. "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 57,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "             predicted     \n",
      "actual   poisonous   edible\n",
      "poisonous     1142       39\n",
      " edible          0     1257\n"
     ]
    }
   ],
   "source": [
    "print_conf_mtx(y_test, y_predict5)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "#### Precision/Recall\n",
    "We can tell that precision has decreased by 0.002 points in this model. This is not a favorable outcome as this would mean two more people could potentially get sick from eating a mushroom that is predicted edible and was not. "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 58,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Precision: 0.970\n",
      "Recall: 1.000\n"
     ]
    }
   ],
   "source": [
    "print(\"Precision: {:.3f}\".format(getPrecision(y_test, y_predict5)))\n",
    "print(\"Recall: {:.3f}\".format(getRecall(y_test, y_predict5)))"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "#### Conclusion Model 5: \n",
    "\n",
    "We saw an equal accuracy in feature selection as in the previous one using only 3 features instead of 5. However the actual accuracy when making predictions seem to be marginally lower than the Naive bayes model. The precision value decreased a ittle, so 2 more mushrooms that are correctly classified in the previous model are wrondly classified as edible in this classifier."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Logistic Regression"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Model 6 - Logistic Regression - All Features"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "For the sixth and seventh model we are going to look into Logistic regression nd how it compares to the other two types of models, Naive bayes and Tree Classifier."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "#### Accuracy\n",
    "\n",
    "The baseline accuracy is at 0.48 which means that for half of the predictions there is a close to 50% chance to predict correct.\n",
    "\n",
    "After training the model with all the features, we get 100% accuracy. Like we have seen in the other models it seems we have features in the data that makes it easy to predict on all features. This is however a slightly better accuracy than what we have seen for all features in the previous models using all features."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 59,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Baseline accuracy: 0.48\n",
      "Logistic regression accuracy: 1.0\n"
     ]
    }
   ],
   "source": [
    "X = df1.values\n",
    "y = df[\"class\"].values\n",
    "\n",
    "X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)\n",
    "\n",
    "log = LogisticRegression()\n",
    "log.fit(X_train, y_train)\n",
    "print(\"Baseline accuracy: {:.2f}\".format(1-y_train.mean()))\n",
    "print(\"Logistic regression accuracy: {:.2}\".format(log.score(X_test, y_test)))"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "#### Confusion Matrix\n",
    "\n",
    "Looking at these values this model has very good precision and recall. only one single person will potentially get sick from eating a poisonous mushroom that was classified as edible. This of course is very unfortunate for this person, but it seems like this is the best case scenario for out prediction models.  "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 60,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "             predicted     \n",
      "actual   poisonous   edible\n",
      "poisonous     1180        1\n",
      " edible          0     1257\n"
     ]
    }
   ],
   "source": [
    "y_pred = log.predict(X_test)\n",
    "print_conf_mtx(y_test, y_pred)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "#### Precision / Recall\n",
    "\n",
    "When looking at the precision and recall values for this model, we can see that when rounding to three numbers we get a value of 1 for both precision and recall which is very very good. "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 61,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Precision: 0.9992\n",
      "Recall: 1.0000\n"
     ]
    }
   ],
   "source": [
    "print(\"Precision: {:.4f}\".format(getPrecision(y_test, y_pred)))\n",
    "print(\"Recall: {:.4f}\".format(getRecall(y_test, y_pred)))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 62,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "F1 score: 0.9996\n"
     ]
    }
   ],
   "source": [
    "print(\"F1 score: {:.4f}\".format(f1_score(y_test, y_pred)))"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "#### Precision / Recall Curve\n",
    "\n",
    "The precision recall curve of this model looks very good. And that goes well with the accuracy we saw when calculation it earlier and the precision recall values. It shows us that we can accheive a 1 for recall as well as for precision at the same time which is what we acheived with the model."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 63,
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "/home/thora/anaconda3/lib/python3.7/site-packages/sklearn/metrics/classification.py:1439: UndefinedMetricWarning: Recall is ill-defined and being set to 0.0 due to no true samples.\n",
      "  'recall', 'true', average, warn_for)\n"
     ]
    },
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "y_probs = log.predict_proba(X_test)\n",
    "y_probs = y_probs[:,1]\n",
    "precisionRecallCurve(y_probs, y_test)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "#### ROC curve\n",
    "\n",
    "The ROC curve is a probability curve for different classes. ROC tells us how good the model is for classifying in these classes. The ROC curve has False Positive Rate on the X-axis and True Positive Rate on the Y axis. The area between the ROC curve and the axis is AUC. A bigger area signifies a better model for distinguishing between classes. The best posible value for ROC is 1, that would mean it should hug the top left corner of the graph. \n",
    "Looking at the curve below, it seems like the model is doing very well. The AUC is 1 which is very very good."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 64,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "AUC: 1.00\n"
     ]
    }
   ],
   "source": [
    "auc = roc_auc_score(y_test, y_probs)\n",
    "fpr, tpr, thresholds = roc_curve(y_test, y_probs)\n",
    "plt.plot(fpr, tpr, color='red', label='ROC')\n",
    "plt.plot([0, 1], [0, 1], color='black', linestyle='--')\n",
    "plt.ylabel('True Positive Rate')\n",
    "plt.xlabel('False Positive Rate')\n",
    "plt.title('ROC curve for Model 6 All features')\n",
    "plt.legend()\n",
    "plt.show()\n",
    "print('AUC: %.2f' % auc)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 65,
   "metadata": {},
   "outputs": [],
   "source": [
    "te_errs = []\n",
    "tr_errs = []\n",
    "tr_sizes = np.linspace(100, X_train.shape[0], 10).astype(int)\n",
    "for tr_size in tr_sizes:\n",
    "  X_train1 = X_train[:tr_size,:]\n",
    "  y_train1 = y_train[:tr_size]\n",
    "  \n",
    "  log.fit(X_train1, y_train1)\n",
    "\n",
    "  tr_predicted = log.predict(X_train1)\n",
    "  err = (tr_predicted != y_train1).mean()\n",
    "  tr_errs.append(err)\n",
    "  \n",
    "  te_predicted = log.predict(X_test)\n",
    "  err = (te_predicted != y_test).mean()\n",
    "  te_errs.append(err)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "#### Learning Curve\n",
    "\n",
    "When looking at the plot below for the learning curves it initially looks like there is a big gap between the curves, but looking at the error values it is evident that these curves are actually very close together. "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 66,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 720x360 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "plt.figure(figsize=(10,5))\n",
    "sns.lineplot(x=tr_sizes, y=te_errs);\n",
    "sns.lineplot(x=tr_sizes, y=tr_errs);\n",
    "plt.title(\"Learning curve Model 6\", size=15);\n",
    "plt.xlabel(\"Training set size\", size=15);\n",
    "plt.ylabel(\"Error\", size=15);\n",
    "plt.legend(['Test Data', 'Training Data']);"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "#### Conclusion Model 6: \n",
    "\n",
    "This model had a better accuracy than the previous models we have made with all features from the data set. However the precision was a little bit lower. When it comes to the learning curve it seems to have both the sets very close together in error and they do not really move toward or away from each other. "
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Model 7 - Logistic Regression - The Most Usable Prediction Model"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "In this model we are looking to make a useful predictor using predictors that are easy for people to identify. We will be using Logistic regression and compare it to the other models we have made so far."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 67,
   "metadata": {},
   "outputs": [],
   "source": [
    "X = df3.values\n",
    "X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 68,
   "metadata": {},
   "outputs": [],
   "source": [
    "def bestLogFeatures(num):\n",
    "    remaining = list(range(X_train.shape[1]))\n",
    "    selected = []\n",
    "    n = num\n",
    "    while len(selected) < n:\n",
    "        min_acc = -1e7\n",
    "        for i in remaining:\n",
    "            X_i = X_train[:,selected+[i]]\n",
    "            scores = cross_val_score(LogisticRegression(), X_i, y_train,\n",
    "           scoring='accuracy', cv=3)\n",
    "            accuracy = scores.mean() \n",
    "            if accuracy > min_acc:\n",
    "                min_acc = accuracy\n",
    "                i_min = i\n",
    "\n",
    "        remaining.remove(i_min)\n",
    "        selected.append(i_min)\n",
    "        print('num features: {}; accuracy: {:.2f}'.format(len(selected), min_acc))\n",
    "    return selected"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "#### Forward Feature Selection\n",
    "\n",
    "We are continuing to use the same features we have used to make a useful predictor earlier. That is cap surface, odor, and cap shape. Looking at the output from forward feature selection we can tell that we hit a cap of 0.99 after 3 features are chosen."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 69,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "num features: 1; accuracy: 0.89\n",
      "num features: 2; accuracy: 0.94\n",
      "num features: 3; accuracy: 0.99\n",
      "num features: 4; accuracy: 0.99\n",
      "num features: 5; accuracy: 0.99\n"
     ]
    }
   ],
   "source": [
    "selected = df3.columns[bestLogFeatures(5)]"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "The best three features are the same as we found in the second Tree classifier. Odor of almond, anis and no odor at all."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 70,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "The best 3 features are: ['odor_n' 'odor_a' 'odor_l']\n"
     ]
    }
   ],
   "source": [
    "predictors = selected[0:3].values\n",
    "print(\"The best 3 features are:\", predictors)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "#### Accuracy\n",
    "\n",
    "We get a very high accuracy for this model as well. It seems to be the exact same as we got for the second Tree classifier. "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 71,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Baseline accuracy: 0.481\n",
      "Logistic regression accuracy: 0.984\n"
     ]
    }
   ],
   "source": [
    "X = df1[predictors].values\n",
    "X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.3, random_state=42)\n",
    "\n",
    "log2 = LogisticRegression()\n",
    "log2.fit(X_train, y_train)\n",
    "print(\"Baseline accuracy: {:.3f}\".format(1-y_train.mean()))\n",
    "print(\"Logistic regression accuracy: {:.3}\".format(log2.score(X_test, y_test)))"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "#### Confusion Matrix \n",
    "\n",
    "The confusion matric is also the same as in the last Tree Classification. There are still 39 mushrooms that are classified as edible, when they are actually poisionous."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 72,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "             predicted     \n",
      "actual   poisonous   edible\n",
      "poisonous     1142       39\n",
      " edible          0     1257\n"
     ]
    }
   ],
   "source": [
    "y_predict7 = log2.predict(X_test)\n",
    "print_conf_mtx(y_test, y_predict7)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "#### Precision/Recall\n",
    "\n",
    "The precision and recall values reflect what we read off the confusion matrix. The recall comes out to one and the precision is a bit lower than we would like it to be."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 73,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Precision: 0.970\n",
      "Recall: 1.000\n"
     ]
    }
   ],
   "source": [
    "print(\"Precision: {:.3f}\".format(getPrecision(y_test, y_predict7)))\n",
    "print(\"Recall: {:.3f}\".format(getRecall(y_test, y_predict7)))"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "#### F1 Score\n",
    "The F1 score reflects a very high precision and recall. The best value one can obtain here is 1. A score of 0.98 is pretty good."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 74,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "F1 score: 0.9847\n"
     ]
    }
   ],
   "source": [
    "print(\"F1 score: {:.4f}\".format(f1_score(y_test, y_predict7)))"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "#### ROC curve\n",
    "\n",
    "In this ROC curve we can tell that the area under the curve comes out to 0.99. This is a very high score and reflects that we have a very high true positive rate and a low false positive rate."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 78,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "iVBORw0KGgoAAAANSUhEUgAAAYkAAAEcCAYAAAAydkhNAAAABHNCSVQICAgIfAhkiAAAAAlwSFlzAAALEgAACxIB0t1+/AAAADh0RVh0U29mdHdhcmUAbWF0cGxvdGxpYiB2ZXJzaW9uMy4xLjAsIGh0dHA6Ly9tYXRwbG90bGliLm9yZy+17YcXAAAgAElEQVR4nO3deXgUVdbA4V93QlgDKsQNkZ3jyiCKOy64oygqo6IiCoKgoCyCy4Ag7iKCICoIIoIgI644fuqgg+I6KjIq6hFFFkEhhCyEPUl/f1QF25hOOqG7q5fzPk8e0tXVXecmoU7fe6vO9QUCAYwxxpjy+L0OwBhjTPyyJGGMMSYkSxLGGGNCsiRhjDEmJEsSxhhjQrIkYYwxJiRLEsaUISI+EZkhIrki8l+v4wlFRK4VkQ/D3PdZEbk32jHticraIyKLROT6WMZkIN3rAEx8EJGVwH5AMVAIvAUMUNXCoH1OBO4FOgAlwAfAbar6XdA+9YExwCXAPsDvwBvAvaq6MRZtiYCTgbOAg1R1y56+mYg0A34BvlLV9kHbGwHrgHWq2mxPj1NdInIncGfQpjSgJrBveb8zEQkArVX1p6Bto4FWqnp1lMM1MWY9CROsi6rWA9oBRwF3lD4hIicA7wCvAQcCzYH/AR+JSAt3nwzgXeBw4FygPnAikAMcG62gRSTSH3aaAiurkyAqiaWuiBwR9PhKnOThKVW9X1XrlX4BDwGLEiipmyiynoT5C1X9XUTexkkWpR4GnlPVx4K2jRCRo4HRwDXu18HA6UE9kA3APaGOJSKHAxOAo4FdwGOqer+IPAv8qqoj3P1OA2ar6kHu45XAk8BVzkO5B2inqt2C3vsxwKeqN4tIA+BRoDNOL2gGMEpVi8vE0xuYDNQQkUJgnKqOEpE+wG04vaMPgX6qus59TQAYAAzC+T/VPERzZwE9gWHu42uA54A+Qcc/1G1XO2AtcIeqvu4+19CN+zTgB+DtMrEfAkxyf5bZwEhV/WeIWMolIj6gB05vsFrcHtKzOD2yEmAZcKqqlojI7Tjt3RdYA/xDVV8JerlPRCbh/Gx+A25S1XdDHKcXzs9yf+C/QF9VXVXduE35rCdh/kJEDgLOA35yH9fB6RG8WM7u/8QZmgE4E3greIiqkuNkAgtxhrYOBFrh9ETC1R04H9gL5wTc2R3uQkTSgMuAOe6+M4Ei9xhHAWcDfxnfVtXpQD/gE/eT9SgR6QQ84L7fAcAq4IUyL+0KHAccVkG8s4ErRCTNTQaZwGelT4pIDWABTo9tX2Ag8LyIiLvLZGC7G0Mv96v0tXWBf7vt3df92TzhJuGq6Igz7PhSFV8XbCjwK5DlvtedQGn9n5/dYzQA7gZmi8gBQa89DlgBNAJGAS+LyD5lDyAiXd33vcQ9zmJg7h7EbEKwnoQJ9qr7qbge8B7Of1JwPj37cT7ZlfUbzn9ogIbAl1U43gXA76o6zn28naCTZhgmquoa9/tVIrIE52T9HNAJ2Kqqn4rIfjhJby9V3QZsEZHxQF9gShjHuQp4RlWXAIjIHUCuiDRT1ZXuPg+o6qZK3udXQHGS6elunMGOx/nZP6iqJcB7IvIG0N3tKV0KHOkOg30rIjOBU9zXXoAzRDbDfbxERF4CuuF8kg9XT2B+uIk+hF04iaypO2+xuPQJVQ3+oDHP/VkeizOMCU7Pc4KqBtznh+J8EJhV5hg34PzMvwcQkfuBO0WkqfUmIsuShAnWVVUXisipOJ9IGwF5QC7OsMEBOMMcwQ4ASseuc9zH4WqC88myutaUeTwH5xP0czjj/aW9iKZADeC3Pz6U4y/n9aEcCCwpfaCqhSKSAzQGVoaIJZTngGtxemanAK3LHGeNmyBKrXKPk4Xz/3VNmedKNQWOE5G8oG3p/PXkGpKI1Ab+DlxUya7FOD/PYDVwkgPAWJwhyHfcn/dUVX3QPcY1wBCgmbtvPf74kAGw1k0QpVbh/FzKago8JiLjgrb5cH5WliQiyJKE+QtVfd+dE3gEJ3FsEZFPcE4g/ymz+2X8MUS0ELhXROqGOem7BuekXp4tQJ2gx/uXs0/ZEsYvAuPc4bKLgROCjrMDaKSqRWHEVdY6nJMSsHtopyHOnEGoWEJ5CXgc+FJVV4lIcJJYBzQREX9QojgY+BFnjqEIJ7H+EPRcqTXA+6p6FtV3CbAJWFTJfqtxTvLfB21r7saJqm7GGXIa6g53/UdEPscZvnwaOANnOK9YRJbinNxLNRYRX1CiOBh4vZwY1gD3qerz4TfPVIclCRPKBGCliLRT1aXA7cDbIvIDzuRpOs6J4AScS2LB+dR6A/CSiAzCOWns7W5bqqpvljnGG8Cj7r5PAhnAYar6GbAU5yRzr7t9UGUBq2q2iCxy4/uldChCVX8TkXdwEshInEt8m+Nc4vp+GD+LOcALIjIH58R4P/BZ0FBT2NyE2wmnd1bWZzjJcbj7CfkkoAvQwT2hvgyMdidsm+EMDZXG8AbwoIj04I/5knZAYenPIQw9cS5OqCzhzcO5aOEbnMTWyY3zBAARuQAnkf0MFOD0PIqBujjJNNvd7zrgiDLvvS9ws4g8gTN0eChQ9u8G4CngHhFZqqrL3AsTzi4znGUiwCauTblUNRtnaGSk+/hD4BycT5u/4XTpjwJOVtXl7j47cMbbf8CZRC3AueqkEeXMNbifOM/COcH8DizHGasHJ+H8D+ck+A7OiSkcc9wY5pTZfg1OsvkO5wQ9nzCHxtyra0bi9AJ+A1oCV4QZT3nv94Wq/mWYTVV3AhfizJ9sBJ4ArlHV0p7DAJzhmd9xrh6aEfTazTiT8VfgnLh/x7mUtWY4MYlIY5yTfdl5kvKMAT7GucorF+fKt6tU9Vv3+dY4vcpC4BPgCVVd5N5PM87dth44EviozHt/5r5+I3Af0E1Vc8oG4F4R9RBO8i4AvsX5uZkI89miQ8YYY0KxnoQxxpiQLEkYY4wJyZKEMcaYkCxJGGOMCSmZLoGtiXMp5m84l9sZY4ypXBrOlX6f49xP9CfJlCQ6EHT7vzHGmCrpiHNZ858kU5L4DSA3dwslJVW/rLdhw3rk5OxJuZrEY21ODdbm1FDdNvv9Pvbeuy6UX5stqZJEMUBJSaBaSaL0tanG2pwarM2pYQ/bXO4wvU1cG2OMCcmShDHGmJAsSRhjjAkpJnMSIvIIzoIpzXAWTfm2nH3SgIk4ayMHcBZemRaL+IwxxpQvVj2JV3EWWKloMZCrcJaWbI1Tcni0iDSLfmjGGGNCiUlPwi0zTdCqYOW5HHjaXWwlW0RexVnkZmz0I6yCkpLK90kUJSXJ1Z5wWJtTQyq2OUri6RLYg/lzT2M1zipccaPGewtpcPVl+Iqqs7hZfMryOgAPWJtTQyq0uQRnqb/8tDSGL14Mrcqu4bTn4ilJRETDhvWq/dqsrMyKd1i1HIqKYORISE+6H50xJoEsz8mhz+uv8/7KlZzbsiW3ipC1TyXnsGqIpzPdapx1hD93H5ftWYQlJ6ewWjeUZGVlkp29ucJ96q7bQO30dDYOuBV8vgr3TQThtDnZWJtTQzK3uaSkhCeemMTDU6eSkVGT8eMf58ore+Dfp3612uz3+yr8cB1PSeJFoI+7jm9DnPVtT/E2pD/z5eURaNAgKRKEMSYx+Xw+Fi9exGmnncHDDz/K/vuHtQpvtcXk6iYRmSgivwIHAQtFZJm7/U0ROcbdbRawAmed40+BMaq6IhbxhctXkEdJg728DsMYk2J27NjB2LEPsGbNanw+HzNmPM/MmXOiniAgdlc33QzcXM72zkHfFwP9YxFPdflLexLGGBMjX3zxXwYPHoDqD2RmZtKv3wDq1KkTs+PbHddV4CvIJ2A9CWNMDGzZsoWRI+/g/PPPorCwkLlz59Ov34CYx2FJogp8eTbcZIyJjfHjxzJlymSuvbY3H3zwKWeccbYnccTTxHXc8+dbT8IYEz35+Xnk5GykRYtW3HzzYM4882yOP/5ET2OynkS4AgF8+TYnYYyJjv/7v39x8snH0rdvLwKBAPXrN/A8QYAlifBt24Zv1y4bbjLGRNSGDRvo0+daevbsTqNGWYwb9xi+OLrM3oabwuTPzwOwnoQxJmKWLfuWSy45ny1btnDnnXdx0023UKNGDa/D+hNLEmHy5ecDliSMMXuuuLiYtLQ02rQROnfuQv/+A2nTpsICqJ6x4aYwlSYJG24yxlRXSUkJzzzzNKecchz5+XnUqFGD8eMfj9sEAZYkwubPzwWsJ2GMqZ6ff15O166duf32oRx4YGO2bdvmdUhhsSQRpt3DTXtZT8IYE77i4mImThzPaaedyPfff8fEiU/yz3++GpOSGpFgcxJh8rkT1yX1LUkYY8Ln9/v5+OPFnHnmOTz44CPst9/+XodUJZYkwuS3iWtjTJh27NjBhAmPcOWVPWjS5GBmzHie2rVrex1WtdhwU5h8eXkE6tSFOLs8zRgTX/7738/o1Okkxo17iDffXACQsAkCLEmEzVeQT4nNRxhjQigsLOTOO4fRpcvZbNu2jRdeeJkbbrjJ67D2mCWJMFmZcGNMRcaPH8v06VPp3bsvH3zwKZ06nel1SBFhcxJh8hXk2z0Sxpg/ycvLJSdnIy1btuaWW4ZwzjmdOfbY47wOK6KsJxEm60kYY4ItWPAaJ53U4U8F+ZItQYAlibDZgkPGGID169fTq1cPevfuwf77H8CECZPjqiBfpNlwU5icBYesJ2FMKvv222+45JLz2bZtGyNGjKZ//4FxV5Av0ixJhKO4GP/mAgL1LUkYk4qKiopIT09H5BC6dOlK//4DadWqtddhxYQNN4XBt7kAsJIcxqSakpISpk17ipNP7rC7IN+4cRNTJkGAJYmw+PLckhw2J2FMyli+/EcuvPBc7rxzOE2bNmP79u1eh+QJSxJh8BeUluSwJGFMsisuLmbChEc4/fQT+fHHH5g06SleeOHlhKu5FCk2JxGG0p6EXQJrTPLz+/18+unHnHNOZx544BH23Xdfr0PylCWJMNiCQ8Ykt23btjFhwliuuqonBx/cNKEL8kWaDTeFwda3NiZ5ffrpJ3TqdBLjxz/C22+/CSR2Qb5IsyQRBltwyJjkU1i4mdtvH8qFF57Drl27ePHF1+jTp7/XYcUdSxJh8OXnEUhLI1C3ntehGGMiZPz4R5gxYxo33HAj77//KaeeerrXIcUlm5MIgz/frduUxLfeG5MKNm3KYdOmTbRq5RTkO++88znmmGO9DiuuWU8iDL78PLvb2pgEFggEWLDgVU4++VhuuOGPgnyWICpnSSIMvnwrE25Molq//neuu+5qeve+hsaND+Kxx55I6oJ8kRaz4SYRaQPMBBoCOcA1qrq8zD77AjOAJkAG8B5ws6oWxSrO8vjzrQKsMYno22+/4eKLz2fHju2MHDmG/v0HkJ5uo+xVEcuexFPAZFVtA0wGppSzz53A96raFjgSOBq4JHYhls+Xb2tJGJNIdu3aBYDIIVx00SX85z8fMXDgIEsQ1RCTJOH2ENoDc91Nc4H2IpJVZtcAkCkifqAmTm9ibSxirIgv39a3NiYRFBcXM3XqExx66KHk5eVSo0YNHnlkAi1bpk5BvkiLVVptAqxV1WIAVS0WkXXu9uyg/e4BXgJ+A+oCj6vqR1U5UMOG1b9MNSsrs/wn8vOovX8WtUM9n8BCtjmJWZuT03fffUfv3r359NNP6dy5M5mZGSnR7mDRaG+89b3+DnwNnAFkAv8nIt1UdX64b5CTU0hJSaDKB87KyiQ7e/Nfn9i+nawdOyjMqMO28p5PYCHbnMSszcmntCDf+PFjqVevHk888TT9+vVm48bCpG53WdX9Pfv9vgo/XMdqTmIN0FhE0gDcfw90twcbCDyvqiWqmg+8Bnh6h8vukhx2Cawxccnv9/Pll59z/vldWLz4c7p1u9yuXoqgmCQJVd0ALAW6u5u6A1+panaZXX8BzgUQkQzgTODbWMQYipXkMCb+bNu2jfvuu5tVq1bi8/l45pnZTJkyg6ysstOcZk/F8uqmfsBAEfkRp8fQD0BE3hSRY9x9BgEdReQbnKTyI/B0DGP8iz8WHLKehDHx4OOPP+S0007gscfG8e9/vwVArVq1PI4qecVsTkJVfwCOK2d756DvfwbOilVM4fAXlFaAtZ6EMV7avLmAMWNGMXPmdJo2bcZLLy2gY8dTvQ4r6VW5J+Fezpoy/lhwyJKEMV6aMGEcs2bNoF+/ASxa9IkliBgJqychIg2AScBlQDFQV0S6AMeo6qgoxuc5W3DIGO/k5OSQk7ORNm2EQYOG0rnzBRx9dAevw0op4fYkngR2AK2Bne62z/hjIjpp7V7fun59jyMxJnUEAgFeeWU+J598DP37X08gECAzs74lCA+EmyTOBG5S1TU4d0WXXrG0X7QCixe+vDwCtWtDzZpeh2JMSvjtt3X07NmdG27oRdOmzXj88Sl2SauHwp24LgD2AX4v3SAiTYD10QgqnvgKrAKsMbHyzTdf07VrZ4qKdnH33ffTt29/0tLSvA4rpYXbk3gGeFFEOgJ+EemAU621vCJ9ScWfZ8X9jIm20oJ8hxxyKJde+ncWLfqE/v0HWIKIA+EmiQdw7n6eDtQC5gBvAeOjFFfc8BVYmXBjoqW4uJgnn3ycE088endBvocfHk/z5i28Ds24wh1uaqiqjwCPBG8UkUbAxohHFUd8eXmU7L+/12EYk3S+//47Bg++iSVLvuTss89l585dXodkyhFuT2JFiO0/RiqQeGULDhkTWcXFxYwd+wBnntmRVatWMmXKM8yaNY99902pW7ASRrhJ4i+XFohIPaAksuHEH1twyJjI8vv9LF26hC5durJ48edcfHE3u3opjlU43CQiv+Bc8lpbRMr2JhrhrP2QvEpK7OomYyJg69atjB37AD179qJZs+Y888xsatpl5QmhsjmJ63F6Ea8DfYK2B4D1qrosWoHFA9/mAnyBgA03GbMHPvzwAwYPHsCqVSs56KAm9O7d1xJEAqkwSajquwAisr+qFsQmpPixuySHlQk3psoKCvK5++6RzJr1LM2bt+DVV9/kxBNP9josU0VhXd2kqgUicgTQEWeYyRf03JgoxeY5W3DImOp77LFHef7557jpplsYNuwO6tSp43VIphrCLfDXG6fA37s4pbz/jbPE6ILoheY9W3DImKrZuHEjmzbl7C7I16XLRbRr197rsMweCPfqptuBzqraBdjm/nsZsCVqkcWB3cNN1pMwpkKBQICXXvrnXwryWYJIfOEmif1UdZH7fYmI+IF/AV2jElWc2D3cZJfAGhPS2rW/cvXVl9G///U0b96CyZOn2iWtSSTcO65/FZGmqroKWA6cj3OndVLfImnDTcZU7Jtv/sdFF3WmpKSYe+55gOuv72f1lpJMuEliHHAEsAq4F3gRqAEMiVJcccGXn0vA5yNQL9PrUIyJKzt37iQjI4NDDjmMyy67gn79BtCsWXOvwzJRENZwk6pOV9V/ud+/AeyNU89pYjSD85pTkqMB+Ku8yqsxSamoqIjHH3+ME088mtzcTdSoUYMHHxxnCSKJVevsp6rbgXQReSDC8cQVX14egfo21GQMwLJl39K58xmMGTOSww8/kqKiYq9DMjFQ6XCTiPQE2uHMRUwF6gAjgX7Ax1GNzmO+gny7kc6kvNKCfBMnPspee+3NtGkz6dKlq01Op4jKajc9DPTASQbdgeOBE4AvgZNV9X9Rj9BDtuCQMU5Bvm+//ZpLLvk7Y8bczz77NPQ6JBNDlfUkrgBOUdXlInIosAzorqrzoh+a93wF+ZTsZ2tJmNSzZcsWHn74fq677vrdBfkyMjK8Dst4oLI5ib1UdTmAqn4PbE2VBAHugkPWkzAp5v33/8Opp57Ak09O4r33FgJYgkhhlfUkfCLShD9qNRWVeYyqro5WcF7z29KlJoXk5+cxatQ/mDNnFi1atOT119/i+ONP9Dos47HKkkRdYCV/XnRoVdD3ASA575zZsQPftm02J2FSxsSJ45k3bw433zyEoUNvo3bt2l6HZOJAZUmiRkyiiEO76zZZT8IksQ0bNrBpUw6HHHIogwYN5aKLLqZt23Zeh2XiSGXrSaTshdD+Arckh/UkTBIKBAL8859zGTnydpo0acrChR+QmVnfEoT5C7uVOARfXi5gScIkn19/XUP37pcycGA/WrcWnnpqut3zYEIKt3bTHhORNsBMoCGQA1xTeuVUmf0uw7lZz4cz53Gmqq6PVZylfAU23GSSz9dfL+WiizoTCAS4//6H6dWrL34rO2MqEMu/jqeAyaraBpgMTCm7g4gcA4wGzlLVI4CTgfwYxribP6+0TLglCZP4duzYAcBhhx3BlVdezQcffMr11/ezBGEqFfZfiIiki8gJItLNfVxbRMK6/EFE9gXaA3PdTXOB9iKSVWbXwcAjqvo7gKrmu3WiYs4mrk0yKCoq4sEHH9xdkC89PZ377nuYgw9u6nVoJkGEu3zp4cBr7sP9gfk4y5dehVOuozJNgLWlE+GqWiwi69zt2UH7HQb8IiIfAPWAl4H7VDUQTpyRZAsOmUT3zTdfM3jwAL7+einnn38hxcUlXodkElC4cxJPAveq6rMikutuW4QzhBTpeNrirKOdAbwFrAaeC/cNGjasV+2DZ2UFrRuxaxvUqkVWk7KdneTypzaniGRvc3FxMaNGjeKhhx6iYcOGzJ8/n0svvdTrsGIu2X/P5YlGm8NNEkfiTDqDM5mMqhaKSJ0wX78GaCwiaW4vIg040N0ebBUwX1V3ADtE5DXgWKqQJHJyCikpqXrHIysrk+zszbsf1/ttAxn1G7ApaFuyKdvmVJAKbQ4EAnzxxRIuvfQyxoy5nzZtmiZ9m8tKhd9zWdVts9/vq/DDdbhzEquAo4I3uJPMP4fzYlXdACzlj6Gp7sBXqppdZtc5wNki4hORGjhDWp5UmvXn59uypSZhFBYWMnLk7fzyywp8Ph/PPDObSZOeYu+99/E6NJPgwk0SdwH/EpGRQIaIDMOZl7irCsfqBwwUkR+Bge5jRORNN+EAvABsAL7DSSrLgOlVOEbEOAsO2XyEiX/vvbeQU045jqlTn+T99/8DQI0aKVsswURYWMNNqvq6iPwG9AE+AgS4TFX/G+6BVPUH4LhytncO+r4EZ91sz9fO9hXkU5KV3PMRJrHl5m7irrvuZN68ObRu3YbXX3+b44473uuwTJIJ9+qmvVX1c+DzKMcTN/z5eRS3bOV1GMaE9PjjjzF//jwGD76VwYOHU6tWLa9DMkko3InrtSKyEHgeeF1Vt0Uxprjgy7dV6Uz8Wb9+PZs25XDooYcxePCtdO16KUce2dbrsEwSC3dOojmwEOdmt/UiMktEznOvUko+gQC+fFvf2sSPQCDACy88T8eOHRgw4AYCgQD16mVagjBRF1aSUNX1qjpRVY8H2gEKPAKsi2ZwXvEVbsZXUkKgviUJ473Vq1dx+eUXc/PN/RE5lClTnrGCfCZmqlPgr4H7lQlsiWw48aG0JIddAmu89vXXS7nwwvPw+Xw8+OA4rr22t9VbMjEV7sR1G5x7G67ESRAvAleo6sdRjM0zPre4X4ldAms8sn37dmrVqsVhhx1Bjx496dv3Rpo0OdjrsEwKCvcjyec48xI3A41VdWCyJggIWnDIehImxnbt2sX48WM54YT2bNqUQ3p6Ovfc86AlCOOZcIeb9vOqGqsXfHlW3M/E3tdfL+WWW25i2bJvuOiiSwjEvKylMX8VMkmISHdVLS3tfZmIlLufqoZdVylR2IJDJpaKi4u5//4xPPHERBo1yuLZZ+fQufMFXodlDFBxT+Ja/lj/oU+IfQJUofheovDb0qUmhvx+P8uXK1dccRWjR99LA/twYuJIyCShqucEfd8xNuHEB19+PgGfz2o3magpLNzMAw/cQ+/eN9CiRUumT59l9ZZMXApr4lpEyi3HISKfRjac+OAryCeQWR/sUkMTBe+++w4dOx7HtGlTWLz4fcAK8pn4Fe5Z8JAQ29tEKpB44s/LsyubTMRt2pTDTTf1pXv3btSrV4833niHnj17eR2WMRWq8OomEXnG/TYj6PtSzYDvoxGU13wF+XaPhIm4yZMn8sor8xkyZDiDBw+jZs2aXodkTKUquwR2bYjvA8CXwLyIRxQH/HlW3M9Exu+//8amTZs47LDDGTz4Vi655O8cfvgRXodlTNgqTBKqOhKcuQdV/VdsQvKeryCfkuYtvQ7DJLBAIMCcObMYNeofNG3ajIULP6BevUxLECbhVHSfxEmq+pH7cLOInFLefqr6QVQi85AvL48S60mYalq58heGDr2FxYsXceKJJ/Poo5OsIJ9JWBX1JKbzx4T18yH2CQBJVy/An59PwK5VN9XgFOQ7F78/jbFjJ9Cjx7VWkM8ktIrukzgk6PsmsQknDuzahW/rFpuTMFUSXJCvZ8/e9O3bn8aND/I6LGP2WLU+4ohIRxE5IdLBxIPSMuG24JAJx86dOxk37iGOP/6o3QX57r77PksQJmmEezPdIhHp6H5/K/Ay8LKI3BbN4Lzgz3dLctglsKYSX331JWeddSoPPXQfxx13vNfhGBMV4fYkjgQ+cb+/ATgNOA64MQoxecoWHDKVKS4u5u67R3LeeWeQm7uJ5557gSlTZrDPPg29Ds2YiAs3SfiBEhFpAaSr6jJVXQ3sE73QvLF7uMmWLjUh+P1+fvllBVdddQ0ffvhfzj23s9chGRM14a4n8TEwATgQeAXATRg5UYrLM/58dy0J60mYIAUF+dx//xj69u1PixatmDZtJunp1Vn915jEEm5P4lpgO6DAKHfbYcCkKMTkqd3DTXZ1k3H9+99v0bHjcTz77HQ+/HAxgCUIkzLC+ktX1WxgeJltbwBvRCMoL/ncnoQtOGQ2btzIiBG38fLLL3LooYcxY8Zs2rc/xuuwjImpsJKEiKQDdwA9gMY4dZxmAQ+q6q7ohRd7/vx8AhkZUKuW16EYjz355CQWLHiVYcPu4JZbhpKRkeF1SMbEXLh95oeAk4BBwCqgKTAC2AsYGp3QvOHLy3Muf7UyCinpt9/WsWnTJg4//AgGDx5Gt26Xc+ihh3kdljGeCTdJXAYcpaob3cfL3IWIlpJsSR4+axEAABwdSURBVKIg326kS0GBQIDZs2cyevQImjVr7hbkq2cJwqS8cCeu04CSMttKgKT7uO3Py7VJ6xTzyy8ruPTSLgwdejN/+1s7pk2baQX5jHGF25OYD7wuIqOA1TjDTXcBL0UrMK/4CvIJ7J10t3+YEP73v6+48MJzSU+vwbhxE7n66p6WIIwJEm6SGIZz6et04ABgHfACcHe4BxKRNsBMoCHO/RXXqOryEPsK8BXwhKreGu4xIsGXl0dxs+axPKTxwLZt26hduzaHH34kvXr1pW/f/hxwwIFeh2VM3An3EtgdwJ3uV3U9BUxW1dkicjUwBehUdicRSXOfe3UPjlVt/gIrE57MduzYwcMP38/zzz/He+99RMOGDRk16h6vwzImblW2xnVrnN7DEcASoJdbjqNKRGRfoD1wlrtpLvC4iGS592AEux3n/ot67lfsBALO1U2WJJLSl19+zq233syyZcvo1u1y/H4bVjKmMpX1JB7HuSfiEeBKnNIcl1TjOE2AtapaDKCqxSKyzt2+O0mISFvgHOB0YGQ1jkPDhtXPK1l1/FBcTJ3G+1EnK7Pa75NIslKgnUVFRQwfPpwJEybQuHFj3njjDc4//3yvw4qpVPg9l2VtjozKksTRQBNV3SYi/wF+iHgELhGpATwNXOcmkWq9T05OISUlgSq/Lisrk5yff6UhsDmtFtuzN1fr+IkkKyuT7BRoZyAQ4Mcff6Znz1489tij7NjhS4l2l0qV33Mwa3P4/H5fhR+uK7sENkNVtwGo6magdpUjcKwBGrvzDaXzDge620sdALQE3hSRlTg37vURkanVPGaV+fLckhx2n0TCy8/PY/jwwaxY8RM+n49p02by8MPjqV+/vtehGZNQKutJ1BSRu4Ie1y7zGFUdU9lBVHWDiCwFugOz3X+/Cp6PcOc6GpU+FpHRQL1YXt3kL3CL+9mCQwntrbfeZPjwwWzYsJ62bdvRokUr0tLSvA7LmIRUWZL4J9A66PH8Mo+rMq7TD5jpJplc4BoAEXkTuEtVv6jCe0VFaU/CyoQnpuzsbP7xj2G8+urLHHro4Tz33FzatWvvdVjGJLQKk4Sq9ojUgVT1B5zV7MpuL3fFFlUdHaljh2t3BVjrSSSkp556nDfffIPbbx/BgAGDrCCfMRFgRfGD7F5wyMpyJIy1a39l06ZNHHlkW4YMGc5ll3VH5BCvwzImaYRbuykl7F5wyHoSca+kpIQZM6bRseNxDB48gEAgQN26dS1BGBNhliSC+PLzKKmXCbbqWFxbseInLr74fG67bQjt2x/D9OnPWb0lY6LEzoZB/Pn5Nmkd55YuXcKFF55LRkZNJkyYTPfuV1uCMCaKwk4SInI6cAWwn6p2FZH2QKaqvh+16GLMl59nQ01xauvWrdSpU4cjj/wbffveyPXX38D++x/gdVjGJL2whptE5EacGk5rcEpmAOwE7otSXJ7w5duCQ/Fmx44dPPjgPRx//FHk5OSQlpbGiBGjLUEYEyPhzkkMBc5U1Xv5Y/Gh74FDoxKVR/z5+daTiCOff/4ZZ5xxMo8+OpZTTjnNCvIZ44Fwk0QmztrW8McNdOk4vYmk4cvPszmJOFBUVMSIEbdxwQVns3XrVl544SUef3wKe9tiUMbEXLhJ4kOgbHmMm4CkmY8Ad7jJ7pHwXHp6OuvWreO6667ngw8+pVOnsyp/kTEmKsKduB4IvCEifYBMEVmG04so927phFRUhL9ws60l4ZG8vFzuuWc0/fsPoFWr1jz99LNWb8mYOBDuynRrReRo4ATgYJwJ7E9K14dICqU30llPIub+9a8F3HbbEHJyNnLUUe1p1aq1JQhj4kTYl8CqagnwkfuVfHJzAavbFEvr16/nzjuHsWDBqxxxRFvmzHmRtm3beR2WMSZIWElCRH4hRMVXVW0R0Yi8srsC7N4eB5I6pk59gnfe+T/+8Y9R3HjjzdSoUcPrkIwxZYTbk7i+zOMDcOYp5kY2HA+5PQkbboquX39dQ27uJo488m8MGTKcK664itat23gdljEmhHDnJN4tu01E3gXexFn3OvGVrkpnE9dRUVqQ7957R9OyZSv+/e/3qVu3riUIY+LcnhT42wYkx1ATWE8iin76aTkXXXQed9xxKx06HMszz8yyekvGJIhw5yTuKrOpDnA+8E7EI/KK9SSi4quvvuTCC8+ldu3aTJz4JJdffqUlCGMSSLhzEq3LPN4CTAaejWg0XsrNJZCeDnXqeB1JUtiyZQt169albdt29O8/kN69b2C//fbzOixjTBVVmiREJA34N/BPVd0e/ZA8kueW5LBPuXtk+/btPProw8ydO5v//OdjGjVqxJ13lu2IGmMSRaVzEu4Nc5OSOkEA5OXZPRJ76LPPPqVTp5OYMOERTj/9DNLT7YY4YxJduBPX/xKR5CnBUZ7cXCvuV01FRUXcccetXHjhOezYsYN5815h4sQn2cvuOTEm4YU7J+EHXhaRD3FKcuy+sU5Ve0UjsJjLswWHqis9PZ3s7Gyuv/4G7rjjLurVq+d1SMaYCAk3SSwHxkYzEM/l5lJyQGOvo0gYubmbuOce507pVq1aM3XqDPx+WzLdmGRTYZIQke6qOldVR8YqIM/k5RGob8NN4Viw4DVuv30oubmb6NDhOFq1am0JwpgkVdn/7CkxicJrgYAzJ2E30lVo/frfue66q+nduwcHHHAgb7+9iO7dr/Y6LGNMFFU23JQa14Nu2wa7dtmNdJWYOvVJFi58mxEj7ubGGweSnh52EWFjTIKq7H95moicTgXJQlXfi2xIsefPdyvAWk/iL1avXkV+ft7ugnxXXnk1LVuWvbfSGJOsKksSNYHphE4SAZKgfpOvdMEhuwR2t+LiYp55Zir33TeG1q3b8M47i6hbt64lCGNSTGVJYkvSrBdRAV9p3Sa7BBaAH39UBg8ewOeff0anTmfyyCOPWb0lY1KUDSoD/oLSBYesJ/HVV1/Spcs51K1bl8mTp9Kt2+WWIIxJYTGbuBaRNsBMoCGQA1yjqsvL7DMSuAIocr/uVNW3IxVDKD5b35rCws3Uq5dJ27btGDBgEL1730BWVpbXYRljPFbhJbCqmhnBYz0FTFbVNjgVZMu7vPa/QAdV/RvQC5gnIrUjGEO5fPmlZcJTr4zEtm3buOeeURx33FFkZ2eTlpbG7bePsARhjAH2bNGhsInIvkB7/ljudC7QXkT+dCZS1bdVdav78GucnkzDaMfn274DgEDNmtE+VFz54IMPOP30E5k0aTxnn30uGRm2xrQx5s9idZtsE2CtW1G2tLLsOnd7KNcAP6vqrzGIz5EiY+9FRUXcdtsQTj31VIqKipk//3XGj3+cBnafiDGmjLicuBaRU4F7gLOq+tqGDatRXK6e04PIysqEunWr/voEtGVLAYMGDeLee++lboq0uVRWViRHURODtTk1RKPNsUoSa4DGIpKmqsXuQkYHutv/REROAGYDF6mqVvVAOTmFlJQEKt8xSO3CHdQDsrM3w9aSqh4yIWzalMPdd49kwIBBtG7dhokTp7Lffg3Izt7M1q2bvQ4vZrKyMp3fcwqxNqeG6rbZ7/dV+OE6JsNNqroBWAp0dzd1B75S1ezg/USkAzAP6KaqS2IRW7ILBAK89trLnHxyB1588QW+/PJzACvIZ4wJSyyHm/oBM0XkLiAXZ84BEXkTuEtVvwCeAGoDU0Sk9HU9VPWbGMaZNH7//TeGDx/CW2/9i3btjuLFF1/n8MOP8DosY0wCiVmSUNUfgOPK2d456PsOsYonFUybNoVFi95l1Kh7ueGGG60gnzGmyuyskWRWrvyF/Pw8/va3o9yCfD1o0aKl12EZYxKUDUwnieLiYqZMmcxpp53ArbcOIhAIUKdOHUsQxpg9Yj2JJPDDD98zePBNfPnlF5x11jmMHTvB6i0ZYyLCkkSCW7LkC7p0OYf69evz1FPTufjibpYgjDERY0kiQZUW5Pvb347illuG0qtXXxo1auR1WMaYJGNzEglm69atjB49gmOPbbe7IN/w4XdagjDGRIX1JBLIRx8tZvDgAaxc+Qs9elxLzZoZXodkjElyliQSgFOQbyizZs2gadNmvPTSAjp2PNXrsIwxKcCGmxJAeno6hYUF9Os3gPff/9QShDEmZqwnEac2btzI6NH/4JZbhtK6dRuefHK61VsyxsScnXXiTCAQ4OWXX6Rjxw688sp8liz5ArCCfMYYb9iZJ46sW7eWHj0up1+/3jRt2oyFCxdz+eVXeh2WMSaF2XBTHJkxYxqLF7/PmDH306dPf9LS0rwOyRiT4ixJeGzFip8pKMinXbv2DB48jCuv7EHz5i28DssYYwAbbvJMcXExTzwxidNPP5FhwwbvLshnCcIYE0+sJ+GB77//jkGDbuSrr5Zwzjnn8fDD463ekjEmLlmSiLHSgnwNGjRg6tQZXHTRJZYgjDFxy5JEjBQU5FO/foPdcw/XXdeHhg0beh2WMcZUyOYkomzLli2MHHkHxx9/FBs2bMDv93PrrbdbgjDGJATrSUTRBx8sYsiQm1m9eiXXXtub2rVreR2SMcZUiSWJKCgqKmL48MHMnj2TFi1a8uqrb3LiiSd7HZYxBiguLiI3N5uiop1ehxJRGzb4KSkpCfl8enoGe++dRVpa1U77liSiID09na1btzBgwCCGDbuD2rVrex2SMcaVm5tNrVp1qFt3/6S6aCQ93U9RUflJIhAIsGVLAbm52TRqdECV3tfmJCIkOzubG2/sw48/KgBPPjmdu+4aYwnCmDhTVLSTunXrJ1WCqIzP56Nu3frV6j1ZkthDgUCA+fPn0bFjB15//RWWLl0CkFJ/gMYkmlT8/1ndNluS2ANr1/7KVVf9nRtv7EPz5i15990Pueyy7l6HZYwxEWNzEntgxoxpfPzxh9x774P07n2DFeQzxlRZt25dyMjIoEaNDIqKdnHFFVfTpUtXAFas+InHH3+MtWvXUFISQEQYOHAI++23/+7Xv/PO/zF37ix27nSGklq1ak3//rew//77l3u8qrIkUUU//7yc/Px82rc/hiFDhtOjx7U0bdrM67CMMQns3nsfokWLVqxY8RO9el3NCSecREZGTQYNuolBg4bRqdOZAMyb9zxDhgxg5swXSE9PZ8GCV5k373keeGAczZs3o6iohCVLvmDTpo0RSxI23BSmoqIiJk2awOmnn8Rttw3dXZDPEoQxJlJatGhFZmZ9srM38NJL8zjqqPa7EwTA5ZdfRd269Vi48G0AZsx4moEDh9CkycG792nf/hgOO+yIiMVkPYkwfPvtNwwadBNff72Uzp278NBD41Jy4suYZFNz3hxqzZ0dlffe3v1qdlRx0bCvv15KgwZ70apVG55/fiZt27b7yz6HHXYEP/20nNzcTWzYsD6iCaE8liQq8eWXn9OlyznstdfeTJ/+HBdccJElCGNMRI0YcRuBQIB169Zy331jqVGjBoFAoMLXVPZ8pFiSCCE/P48GDfbiqKOOZtiwO7j22t7svfc+XodljImgHZdfWeVP+9FQOifx3nsLueeekcyd+zKtWrVh2bJv/rLvd999y8UXd2OffRqSlbUv33+/jGOPPT5qscVsTkJE2ojIJyLyo/tv63L2SRORySLys4j8JCLXxyq+UoWFhYwYcdufCvINHjzMEoQxJuo6dTqTDh2OZ/bsZ7n00stYsuRL3ntv4e7n5817ns2bCzjrrHMB6NmzN5MmPcratb/u3uezzz5h2bJvIxZTLHsSTwGTVXW2iFwNTAE6ldnnKqAV0BpoCHwlIgtVdWUsAly0+H1uHXEbq1evolevPtSpY3dLG2Niq1+/AfTufTVXXdWT8eMnM3nyBJ56ahKBALRu3Ybx4yeTnu6curt2vZSaNWsyYsRwdu7cic/no2XL1tx4480RiycmSUJE9gXaA2e5m+YCj4tIlqpmB+16OfC0qpYA2SLyKvB3YGw04ysqKaY38Mw1V9CyZStef/0tjj/+xGge0hhjAJg/f8GfHh90UBPefvt9ABo2bMT48ZMrfP15513AeeddUGHtpj0Rq55EE2CtqhYDqGqxiKxztwcniYOBVUGPV7v7hK1hw3pVj05asiMjg9tvuYVRY8ZQq1bqlPTOysr0OoSYszanhlBt3rDBT3p6cl79X1m7/H5/lf8Wkm7iOienkJKSKs76n9WFWRs3snE7bN68i82bd0UnuDiTlZVJdvZmr8OIKWtzaqiozSUlJVH5xO21cHoSJSUlf/m5+P2+Cj9cxyqdrgEai0gaOBPUwIHu9mCrgaZBjw8uZ5/I8/vxZabeJy1jjKlMTJKEqm4AlgKl1e+6A1+VmY8AeBHoIyJ+EckCugIvxSJGY0zqiNU9BvGkum2O5cBcP2CgiPwIDHQfIyJvisgx7j6zgBXAcuBTYIyqrohhjMaYJJeensGWLQUplShKFx1KT8+o8mtjNiehqj8Ax5WzvXPQ98VA/1jFZIxJPXvvnUVubjaFhXlehxJRfn94y5dWVdJNXBtjTEXS0tKrvIRnIojWBQrJeR2YMcaYiLAkYYwxJqRkGm5KA+ea3+rak9cmKmtzarA2p4bqtDnoNeUurelLohn+k4HFXgdhjDEJqiPwYdmNyZQkagIdgN+AYo9jMcaYRJEGHAB8Duwo+2QyJQljjDERZhPXxhhjQrIkYYwxJiRLEsYYY0KyJGGMMSYkSxLGGGNCsiRhjDEmJEsSxhhjQkqmshyVEpE2wEygIZADXKOqy8vskwZMBM4FAsCDqjot1rFGSphtHglcARS5X3eq6tuxjjVSwmlz0L4CfAU8oaq3xi7KyAq3zSJyGTAS8OH8fZ+pqutjGWukhPm3vS8wA2gCZADvATeralGMw91jIvIIcCnQDDhSVb8tZ5+In79SrSfxFDBZVdsAk4Ep5exzFdAKaA2cAIwWkWYxizDywmnzf4EOqvo3oBcwT0RqxzDGSAunzaX/oaYAr8YwtmiptM3u4l6jgbNU9QicUjb5sQwywsL5Pd8JfK+qbYEjgaOBS2IXYkS9CpwCrKpgn4ifv1ImSbifKNoDc91Nc4H27jKpwS4HnlbVEnd51VeBv8cu0sgJt82q+raqbnUffo3zKbNhzAKNoCr8ngFuB94AfoxReFFRhTYPBh5R1d8BVDVfVbfHLtLIqUKbA0CmiPhxSvdkAGtjFmgEqeqHqrqmkt0ifv5KmSSB091c665+V7oK3jp3e7CD+XOmXl3OPoki3DYHuwb4WVV/jUF80RBWm0WkLXAOMD7mEUZeuL/nw4AWIvKBiCwRkREikqilUsNt8z1AG5yabr8Db6vqR7EMNMYifv5KpSRhKiEip+L8p+rudSzRJCI1gKeBfqUnmRSRDrQFzgJOBc4DengaUfT9Had3fADQGDhFRLp5G1JiSaUksQZo7I5Dl45HH+huD7YaaBr0+OBy9kkU4bYZETkBmA10VVWNaZSRFU6bDwBaAm+KyEpgENBHRKbGNtSICff3vAqYr6o7VHUz8BpwbEwjjZxw2zwQeN4dfsnHafPpMY00tiJ+/kqZJKGqG4Cl/PEpuTvwlTtuF+xFnBOG3x3f7Aq8FLtIIyfcNotIB2Ae0E1Vl8Q2ysgKp82qulpVG6lqM1VtBkzAGcftG/OAI6AKf9tzgLNFxOf2ps4A/he7SCOnCm3+BedKH0QkAzgT+MtVQUkk4uevlEkSrn7AQBH5EecTRj8AEXnTvfIDYBawAlgOfAqMUdUVXgQbIeG0+QmgNjBFRJa6X0d6E25EhNPmZBNOm18ANgDf4ZxglwHTPYg1UsJp8yCgo4h8g9PmH3GGGhOOiEwUkV+Bg4CFIrLM3R7V85etJ2GMMSakVOtJGGOMqQJLEsYYY0KyJGGMMSYkSxLGGGNCsiRhjDEmJEsSJuGJyGwRGe11HJURERWRjhU8/46IXBXLmIypTEqVCjfxzb37eT8guFRGG1Vd50Ess4HLgJ3u1xfAAFWtdjFAVZWg978XOEhVrw16/uxqBxyCiKQDu4CtOMXu8nCK4d2mqiVhvP5MYJp706FJQdaTMPGmi6rWC/qKeYIIcr+q1sMpkLYJeMbDWPbU4W5bOuHUa+rpcTwmQVhPwsQ9t8zzP3HWP6iFc+dsf1X9vpx99wWeBU4ESoBvVfUU97mDgEnu+xTilM2eXNnxVXWLiMzFWeAGEakFPIxTPK4Ep6TJ7aq6s5Lj/wpcDdQDhgM+t9icqurRIvIhMM19v/XAsar6g/va/XFKTBykqjkiciFOMcamOGUm+pW3CE05bflRRD4G2gX9zK4HhuLcybsBeEBVp4lIA2ABUFNECt3dWwAbccqs9wYaAAtxfh+5lR3fJB7rSZhE8QbOQir745wUZ4XYbxhOWYIsd9+RsLsA3BvA5zjVQM8ChonIGZUdWEQygStxVrADuAs4Bqei6lHAScAdFR0/mKq+gZNknnd7S0eXeX4bzjoAwdV4LwfedRNEB5zSEtfjrPvxDPCaW5uosrYc6sb7U9Dm9cD5QH2gDzBJRNq6BfG6AKuDenYbgCHu/qfgJJYtOKuhmSRkPQkTb14VkdKlJRepald37PzZ0h3cSepsEamrqlvKvH4XToXXg1X1Z+B9d/vxQH1Vvd99/JOITMdZtvXdELHcLiKDgG3AZzir9oGz+lef0mJyIjIGeAy4u4LjV9UcnBPvKPfxle4xAPriLLf6ufv4GRH5B9ABCLVWwtduoqwDPE/QKm6quiBov/dE5F2gI06J7fLcAFyvqmth9+/jJxHpGc48h0ksliRMvOmqqguDN7gntweAbkAjnGEc3O/LJokHcU7W74pIMfCUqo7FGZY5WETygvZNAxZVEMuDqjq6nO0H8OeFXVbh9E4qOn5VLQT2EpGjcSabD8cpcw1OW64SkcFB+2cExVCetjhlpC8H7sVJFjsBROQCnB5Pa5zRhTo4Pa5QDgYWiEhwQggA++Is7GOSiCUJkwiuATrjTLquwhliycZZZvVPVLUAZ5nOwW4l2/+IyH9xauovV9VDIxDPbzgn6tJ1Nw7GXRIz1PFVtWyPosLKmqpaJCIv4gw55QOvBfWa1gB3q+pDVQna/ZQ/V0S6AiOAW921zOfj9Kj+paq7ROQN/vjZlhfnr8CVqvpZVY5vEpMlCZMIMoEdQA7Op9z7Qu0oIl1wSmGvwDm5FrtfnwI7RWQoMBlnWOgwIENVv6xiPHOBu0RkCc7JdCTOgk0VHb+s9TglrH2qGiphzMEp710I3Bq0fSrwooi8h3Npbl2chXTeK2f4rTwPAB+KyEM4P4cMnKRb7PYqznDftzTORiKS6S5UBPAUcL+IXKeqq93J+uNV9fUwjm0SjE1cm0QwA2f94nU4ayB8XMG+AryHc2L9CHjMXUC+CKc3ciywEucKnSk4k7VVdTfOYj3f4Izbf4Zz4g15/HLeYx7OyXmT29Mpz8dAEc4k+DulG91P8P2BJ4FcnDUSrg43eFVdCnwC3KqqeTg9n1dwLvPthjPBX7rvtziL1qwUkTw3ITwKvIUzpLbZjbNDuMc3icXWkzDGGBOS9SSMMcaEZEnCGGNMSJYkjDHGhGRJwhhjTEiWJIwxxoRkScIYY0xIliSMMcaEZEnCGGNMSJYkjDHGhPT/p9DopH/vW6wAAAAASUVORK5CYII=\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "AUC: 0.99\n"
     ]
    }
   ],
   "source": [
    "y_probs = log2.predict_proba(X_test)\n",
    "y_probs = y_probs[:,1]\n",
    "auc = roc_auc_score(y_test, y_probs)\n",
    "fpr, tpr, thresholds = roc_curve(y_test, y_probs)\n",
    "plt.plot(fpr, tpr, color='red', label='ROC')\n",
    "plt.plot([0, 1], [0, 1], color='black', linestyle='--')\n",
    "plt.ylabel('True Positive Rate')\n",
    "plt.xlabel('False Positive Rate')\n",
    "plt.title('ROC curve for Model 7 Usable')\n",
    "plt.legend()\n",
    "plt.show()\n",
    "print('AUC: %.2f' % auc)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 76,
   "metadata": {},
   "outputs": [],
   "source": [
    "te_errs = []\n",
    "tr_errs = []\n",
    "tr_sizes = np.linspace(100, X_train.shape[0], 10).astype(int)\n",
    "for tr_size in tr_sizes:\n",
    "  X_train1 = X_train[:tr_size,:]\n",
    "  y_train1 = y_train[:tr_size]\n",
    "  \n",
    "  log2.fit(X_train1, y_train1)\n",
    "\n",
    "  tr_predicted = log2.predict(X_train1)\n",
    "  err = (tr_predicted != y_train1).mean()\n",
    "  tr_errs.append(err)\n",
    "  \n",
    "  te_predicted = log2.predict(X_test)\n",
    "  err = (te_predicted != y_test).mean()\n",
    "  te_errs.append(err)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "#### Learning Curve\n",
    "\n",
    "This learning curve reflects that the error for the test data decreases rapidly as well as the error for the training data. at around a 1000 instances, the curves flatten out and the error stays relatively stable. The area between them is also very small. Looking at the values of error, these errors are marginal."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 77,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "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ZEgdu3fvzOpYBXEiIiJlTgFc+fO8EJDdLF8FcSIiIiJlSDlxIiIikhNz515ALBYjHo+xYsVyJk3aF4DGRscVV3wj6/P97W9/ZuTIUUydun+7+6+55ussWPAKgwYNZufOHQwbVsecOadxwgn/3OW5X375RZLJJEcccVTW7SoVCuJyrGnTDm773Vt8ds6BDBpYVezmiIiIFMwtt9wJwOrVq/jUp87jjjt+0avz/e1vf2HatIM7DOIAzj//IubMOR2AhQvf5sorv8rmzZs544yzOz33yy+/SCKRUBAne+zanWDhik38491mZk0bXezmiIiIlIxHHvkNv/nNAyQSCWprB3H55V9l/PgGXn11AT/4wXdIpfwlUy68cC4DBlTz7LPPsGDBKzz00IOce+55XY6wNTZO5dJLv8B3vvNtzjjjbJqa1nH11f/N++9vZ/fu3cye/QE+85l/Z9Ei45FHfkMqleL555/lhBNO5PTTz+YrX/kCmzdvZteuXRxwwIF86UtXEImUbqhUui0rU2PqB1JTHcWWb1QQJyIiBfXMa6t5+h/tPmaz12ZNG83Mg3r+d+2VV17iqaf+yo033kY0GuXpp5/kf//3W9xww0+5++7bOffc8zn++BNJpVJs27aN2tpaZsyYybRpB7eOtHXH/vsfSHPzerZs2cygQYO4/vofUl1dTSwW4z//87O8+OLzHHHEUZx88qkkEgkuueRSAJLJJFdddS2DBg0imUzyzW9eye9//winnDKnx+853xTE5VjI83Djh/D28k3FboqIiEjJeOaZJ1m40Jg79wLAXx/t/fffB+DQQw/nzjt/xqpVKzniiKPYf/8De3GlPTM8E4kkN9zwfd544zUAmpvXs2jRwnZvoSaTSe6++w5eeOE5kskEW7Zsoba2thftyD8FcXngGobw8sIm1m/awfAh1cVujoiIVIiZB/VutCyfUqkUH/3ox7joorl77Tv33PM49tgP8NJLz/O97/0vxxwzi09+8jM9us5bb73J8OH1DBo0mNtuu5kdO3Zwyy0/p6qqimuvvZrdu3e1e9xjjz3KW2+9wY033sqAAQO4/fZbWLt2TY/aUChaYiQPpjYMBdBonIiISGDWrH/i979/hPXrmwBIJBK8/fZbACxfvpRx48YzZ87pnH76Wbz11hsADBgwkG3btnX7GosWLeTHP/4/PvEJf7Rv69at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\n",
      "text/plain": [
       "<Figure size 720x360 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "plt.figure(figsize=(10,5))\n",
    "sns.lineplot(x=tr_sizes, y=te_errs);\n",
    "sns.lineplot(x=tr_sizes, y=tr_errs);\n",
    "plt.title(\"Learning curve Model 7\", size=15);\n",
    "plt.xlabel(\"Training set size\", size=15);\n",
    "plt.ylabel(\"Error\", size=15);\n",
    "plt.legend(['Test Data', 'Training Data']);"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "#### Conclusion Model 7\n",
    "\n",
    "Overall in this model we can tell that it has the same performance as the last Tree Classifier we built using the same features. Using feature selecton the same features were pickedd out in the same amount of steps. The biggest difference between these models lie in the learning curves in which the Tree classifier seemed to have a constant error for the training set while the learning curve in this logistic regression model seems to have the training and test data error follow each other more closely."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Conclusion\n",
    "\n",
    "In total we have made 7 models, 3 Naive bayes models, 2 Tree classifiers and 2 Logistic regression models. The first observation we made is that our dataset inherently gives us a very high accuracy and precision/recall values for all of our models. At first we thought this could be because we had an imbalanced dataset but when we went back and looked at the values we are classifying by, Edible/poisonous, there is about 50% of each present in the dataset. \n",
    "\n",
    "\n",
    "The best accuracy we acheived was using Logistic regression on all features of the dataset. This gave us an accuracy of 100%. When it comes to making a most useful classifyer that uses features that people could easily observe, we got predictions using the fewest features (only three variants of odor) when classifying using Logistic regression and Tree Classification. However, the accuracy was actually a little bit higher for our Naive Bayes model using 5 features. The highest accuracy acheived for the \"Useful\" model, was 0.985. This gave us a Precision of 0.972. This means that worst case scenario, 36 people who had their mushroom predicted to be edible, would actually eat a poisonous one. \n",
    "\n",
    "Overall, based on this dataset, it looks like it is fairly easy, to decide if a mushroom is edible, based on cap-shape and smell. The best predictors are smell of nothing, almond or anis and a cap shape of conical or bell. Still, there is no 100% guarantee that you won't be poisoned using these prediction models."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.7.3"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2
}
