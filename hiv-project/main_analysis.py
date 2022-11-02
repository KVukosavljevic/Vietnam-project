# Load NeuroKit and other useful packages
import pathlib
import neurokit2 as nk
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.decomposition import PCA
import seaborn as sns
from sklearn.model_selection import train_test_split

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
from sklearn.model_selection import cross_val_score
import umap

from data import get_file_names, get_datatype, read_data, get_label
from features import get_features

from pathlib import Path 
import typer
import os

# Path to data
sh_path=Path("./RAW_DATA/Shimmer/")
ppg_path=Path("./RAW_DATA/PPG-Smartcare/")

# [2] create a function to calculate accuracy and f beta score for each model
from sklearn.metrics import fbeta_score, accuracy_score, confusion_matrix


def test_predict(clf, X_test, y_test):
    results = {}
    predictions_test = clf.predict(X_test)
    accuracy = results['acc_test'] = accuracy_score(y_test, predictions_test)
    fbeta = results['f_test'] = fbeta_score(y_test, predictions_test, beta=0.5)

    print('test accuracy for {} model is {}'.format(
        clf.__class__.__name__, results['acc_test']))
    print('test F-beta for {} model is {}'.format(
        clf.__class__.__name__, results['f_test']))
    print('------------------------------------------------------------------')

    return results

def unison_shuffled_copies(a, b):
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]


def main(
):
    # Loading the data
    tr_abnormal = np.load('tr_abnormal.npy', allow_pickle = True)
    tr_normal = np.load('tr_normal.npy', allow_pickle = True)

    ts_abnormal = np.load('ts_abnormal.npy', allow_pickle = True)
    ts_normal = np.load('ts_normal.npy', allow_pickle = True)

    print(f"Training data shape normal {tr_normal.shape} and abnormal {tr_abnormal.shape}.")
    print(f"Test data shape normal {ts_normal.shape} and abnormal {ts_abnormal.shape}.")

    # Mix the data
    tr_abnormal = np.concatenate(tr_abnormal)
    tr_normal = np.concatenate(tr_normal)

    ts_abnormal = np.concatenate(ts_abnormal)
    ts_normal = np.concatenate(ts_normal)

    train_data = np.concatenate((tr_abnormal, tr_normal), axis=0)
    train_labels = np.concatenate([np.zeros(len(tr_abnormal)), np.ones(len(tr_normal))])

    test_data = np.concatenate((ts_abnormal, ts_normal), axis=0)
    test_labels = np.concatenate([np.zeros(len(ts_abnormal)), np.ones(len(ts_normal))]) 
    
    print("After concat:")
    print(f"Training data shape normal {train_data.shape} and labels {train_labels.shape}.")
    print(f"Test data shape normal {test_data.shape} and labels {test_labels.shape}.")

    # PCA
    n_components = 2  # number of coordinates
    pca = PCA(n_components=n_components)
    X_reduced=pca.fit_transform(train_data)
    print(X_reduced.shape)

    # Correlation map
    f, ax = plt.subplots(figsize=(15, 15))
    sns.heatmap(pd.DataFrame(train_data).corr(numeric_only=True), annot=True, linewidths=0.5,
                fmt='.1f', ax=ax, cmap='coolwarm')
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    plt.title('Correlation Map')
    plt.show()

    # Data drop
    dropped_att = [2,3,6,11,15,5,13]
    data_red = pd.DataFrame(train_data).drop(dropped_att, inplace=False, axis=1)
    data_red_test = pd.DataFrame(test_data).drop(dropped_att, inplace=False, axis=1)

    f, ax = plt.subplots(figsize=(15, 15))
    sns.heatmap(pd.DataFrame(data_red).corr(numeric_only=True), annot=True, linewidths=0.5,
                fmt='.1f', ax=ax, cmap='coolwarm')
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    plt.title('Correlation Map')
    plt.show()

    ##########################################################################################################

    # Data analysis
    train_data = np.array(data_red)
    test_data = np.array(data_red_test)
    
    print(f"train_labels check: {train_labels.shape} and {train_data.shape}")
    print(f"train_labels check: {test_labels.shape} and {test_data.shape}")

    #X_train, X_test, y_train, y_test = train_test_split(
    #    train_data, train_labels, test_size=0.25, random_state=0)

    X_train, y_train = unison_shuffled_copies(train_data, train_labels)
    X_test , y_test = unison_shuffled_copies(test_data, test_labels)

    print(f'Training set with feature selection has {X_train.shape[0]} samples and type {type(X_train)}.')
    print(f'Testing set with feature selection has {X_test.shape[0]} samples and type {type(X_test)}.')

    #LDA
    clf_lda = LinearDiscriminantAnalysis()
    clf_lda.fit(X_train, y_train)

    print(
        f'Accuracy of {clf_lda.__class__.__name__} on training set: {clf_lda.score(X_train, y_train):.2f}')
    tr_lda = clf_lda.score(X_train, y_train)
    val_lda = test_predict(clf_lda, X_test, y_test)
    print(val_lda)

    #QDA
    clf_qda = QuadraticDiscriminantAnalysis()
    clf_qda.fit(X_train, y_train)  

    print(f'Accuracy of {clf_qda.__class__.__name__} on training set: {clf_qda.score(X_train, y_train):.2f}')
    tr_qda = clf_qda.score(X_train, y_train)
    val_qda = test_predict(clf_qda, X_test, y_test)
    print(val_qda)

    #KNN
    k = 2
    clf_knn = KNeighborsClassifier(n_neighbors=k, algorithm='auto')
    clf_knn.fit(X_train, y_train)
    
    print(f'Accuracy of {clf_knn.__class__.__name__} on training set: {clf_knn.score(X_train, y_train):.2f}')
    tr_knn = clf_knn.score(X_train, y_train)
    val_knn = test_predict(clf_knn, X_test, y_test)
    print(val_knn)

    # Plotting the algorithm results
    fig, ax = plt.subplots(1, figsize=(14, 10))
    scores = np.array([(tr_lda,val_lda['acc_test']),(tr_qda,val_qda['acc_test']),(tr_knn,val_knn['acc_test'])])
    barWidth = 0.3
    bars1 = scores[:,0]
    bars2 = scores[:,1]
    print(bars1)
    print(bars2)

    # The x position of bars
    r1 = np.arange(len(bars1))
    r2 = [x + barWidth for x in r1]
    
    plt.bar(r1, bars1, width = barWidth, color = 'blue', edgecolor = 'black', capsize=7, label='train')
    plt.bar(r2, bars2, width = barWidth, color = 'cyan', edgecolor = 'black', capsize=7, label='test')
    plt.xticks([r + barWidth for r in range(len(bars1))], ['LDA', 'QDA', 'KNN'])
    plt.ylabel('Accuracy')
    plt.title('Algorithm comparison')
    plt.legend()
    
    # Show graphic
    plt.show()

    #UMAP
    mapper = umap.UMAP(n_neighbors=18).fit(X_train, np.array(y_train))
    test_embedding = mapper.transform(X_test)
    train_embedding = mapper.transform(X_train)

    fig, ax = plt.subplots(1, figsize=(14, 10))
    plt.scatter(*mapper.embedding_.T, s=0.3, c=np.array(y_train), cmap='Spectral', alpha=1.0)
    plt.setp(ax, xticks=[], yticks=[])
    cbar = plt.colorbar(boundaries=np.arange(3)-0.5)
    cbar.set_ticks(np.arange(2))
    cbar.set_ticklabels(['Abnormal', 'Normal'])
    plt.title('ECG features Embedded via UMAP Transform')

    fig, ax = plt.subplots(1, figsize=(14, 10))
    plt.scatter(*test_embedding.T, s=0.3, c=np.array(y_test), cmap='Spectral', alpha=1.0)
    plt.setp(ax, xticks=[], yticks=[])
    cbar = plt.colorbar(boundaries=np.arange(3)-0.5)
    cbar.set_ticks(np.arange(2))
    cbar.set_ticklabels(['Abnormal', 'Normal'])
    plt.title('ECG features Embedded via UMAP Transform')

    #LDA
    clf_lda = LinearDiscriminantAnalysis()
    clf_lda.fit(train_embedding, y_train)

    print(
        f'Accuracy of {clf_lda.__class__.__name__} on training set: {clf_lda.score(train_embedding, y_train):.2f}')
    tr_ldaump = clf_lda.score(train_embedding, y_train)
    val_ldaump = test_predict(clf_lda, test_embedding, y_test)
    print(val_ldaump)

    #QDA
    clf_qda = QuadraticDiscriminantAnalysis()
    clf_qda.fit(train_embedding, y_train)  

    print(f'Accuracy of {clf_qda.__class__.__name__} on training set: {clf_qda.score(train_embedding, y_train):.2f}')
    tr_qdaump = clf_qda.score(train_embedding, y_train)
    val_qdaump = test_predict(clf_qda, test_embedding, y_test)
    print(val_qdaump)

    #KNN
    k = 2
    clf_knn = KNeighborsClassifier(n_neighbors=k, algorithm='auto')
    clf_knn.fit(train_embedding, y_train)
    
    print(f'Accuracy of {clf_knn.__class__.__name__} on training set: {clf_knn.score(train_embedding, y_train):.2f}')
    tr_knnump = clf_knn.score(train_embedding, y_train)
    val_knnump = test_predict(clf_knn, test_embedding, y_test)
    print(val_knnump)

    # Plotting the algorithm results

    # Plotting all the algorithm results
    fig, ax = plt.subplots(1, figsize=(14, 10))
    scores = np.array([(tr_lda,val_lda['acc_test']),(tr_qda,val_qda['acc_test']),(tr_knn,val_knn['acc_test'])])
    barWidth = 0.3
    bars1 = scores[:,0]
    bars2 = scores[:,1]
    print(bars1)
    print(bars2)

    # The x position of bars
    r1 = np.arange(len(bars1))
    r2 = [x + barWidth for x in r1]
    
    plt.bar(r1, bars1, width = barWidth, color = 'blue', edgecolor = 'black', capsize=7, label='train')
    plt.bar(r2, bars2, width = barWidth, color = 'cyan', edgecolor = 'black', capsize=7, label='test')
    plt.xticks([r + barWidth for r in range(len(bars1))], ['LDA', 'QDA', 'KNN'])
    plt.ylabel('Accuracy')
    plt.title('Algorithm comparison')
    plt.legend()
    
    # Show graphic
    plt.show()

    fig, ax = plt.subplots(1, figsize=(14, 10))
    scores = np.array([(tr_lda,val_lda['acc_test']),(tr_qda,val_qda['acc_test']),(tr_knn,val_knn['acc_test']),(tr_ldaump,val_ldaump['acc_test']),(tr_qdaump,val_qdaump['acc_test']),(tr_knnump,val_knnump['acc_test'])])
    barWidth = 0.3
    bars1 = scores[:,0]
    bars2 = scores[:,1]
    print(bars1)
    print(bars2)

    # The x position of bars
    r1 = np.arange(len(bars1))
    r2 = [x + barWidth for x in r1]
    
    plt.bar(r1, bars1, width = barWidth, color = 'blue', edgecolor = 'black', capsize=7, label='train')
    plt.bar(r2, bars2, width = barWidth, color = 'cyan', edgecolor = 'black', capsize=7, label='test')
    plt.xticks([r + barWidth for r in range(len(bars1))], ['LDA', 'QDA', 'KNN','LDA UMAP', 'QDA UMAP', 'KNN UMAP'])
    plt.ylabel('Accuracy')
    plt.title('Algorithm comparison')
    plt.legend()
    
    # Show graphic
    plt.show()

if __name__ == "__main__":
    typer.run(main)

