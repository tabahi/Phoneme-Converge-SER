"""
-----
Author: Abdul Rehman
License:  The MIT License (MIT)
Copyright (c) 2020, Tabahi Abdul Rehman
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice,
   this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
POSSIBILITY OF SUCH DAMAGE.


Dependencies: sklearn (v2.1), numba (v0.45.1), pickle, matplotlib
"""

import numpy as np
from numba import jit #install numba to speed up the execution
from timeit import default_timer as timer

from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
#from sklearn.svm import LinearSVC
#from sklearn.svm import NuSVC
#from sklearn.gaussian_process import GaussianProcessClassifier
#from sklearn.gaussian_process.kernels import RBF
#from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
#from sklearn.ensemble import AdaBoostClassifier
#from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
#from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier


'''
Confusion matrix, UAR, WAR calcultions
'''

def create_conf_matrix(expected, predicted, classes):
        n_classes = len(classes)
        m = np.zeros((n_classes, n_classes), dtype=np.uint16)

        for i in range(len(expected)):
            for exp in range(n_classes):
                for pred in range(n_classes):
                    if(expected[i]==classes[exp]) and (predicted[i]==classes[pred]):
                        m[exp][pred] += 1
        return m



def get_UAR_WAR(confusion_matrix):
    class_recalls = np.array([])
    class_weights = np.array([])
    for row in range(0, len(confusion_matrix)):
        if(np.nansum(confusion_matrix[row]) > 0):
            class_recalls = np.append(class_recalls, confusion_matrix[row, row] / np.nansum(confusion_matrix[row]))
        else:
            class_recalls = np.append(class_recalls, confusion_matrix[row, row] * 0)
        class_weights = np.append(class_weights, np.nansum(confusion_matrix[row]) / np.nansum(confusion_matrix))
    uar = np.mean(class_recalls)
    war = np.sum(class_recalls*class_weights)

    return round(uar*100, 2), round(war*100, 2)


'''
Classifier functions and objects
'''

def Test(classifier_model, X_features):

    from sklearn import preprocessing
    X_features = preprocessing.scale(X_features, with_mean=False, with_std=True, axis=0)
    y_pred = classifier_model.predict(X_features)
    #test_acc = accuracy_score(Y_labels, y_pred) # same as WAR
    #results_array.append(test_acc)
    
    
    return y_pred
    

class ClassifierObject(object):

    classifiers = {
            'SVC1': SVC(decision_function_shape='ovr', kernel='rbf', C=10, gamma='scale', probability=False, class_weight='balanced',),
            'SVC2': SVC(decision_function_shape='ovr', kernel='rbf', C=1, gamma='scale', probability=False, class_weight='balanced'),
            'SVC3': SVC(decision_function_shape='ovr', kernel='rbf', C=0.1, gamma='scale', probability=False, class_weight='balanced'),
            'SVC4': SVC(decision_function_shape='ovr', kernel='rbf', C=0.01, gamma='scale', probability=False, class_weight='balanced'),
            'SVC5': SVC(kernel='poly', degree=3, C=1,  gamma='scale', random_state=0, probability=False),
            'RF02': RandomForestClassifier(max_depth=5, n_estimators=50, max_features=5),
            'RF01': RandomForestClassifier(max_depth=10, n_estimators=100, max_features=7),
            'RF01': RandomForestClassifier(max_depth=10, n_estimators=100, max_features=10),
            'RF03': RandomForestClassifier(max_depth=30, n_estimators=100, max_features=7),
            'RF04': RandomForestClassifier(max_depth=30, n_estimators=100, max_features=10),
            'RF04': RandomForestClassifier(max_depth=30, n_estimators=100, max_features=10),
            'RF04': RandomForestClassifier(max_depth=30, n_estimators=100, max_features=10),
            'KNN1': KNeighborsClassifier(n_neighbors=20, weights='uniform', leaf_size=30, p=2, metric='minkowski'),
            'MLP1': MLPClassifier(alpha=1, max_iter=2000, validation_fraction=0.3, hidden_layer_sizes=(400,100,50,), )
            
        }


    def Train(self, X_features, Y_labels):

        from sklearn import preprocessing
        X_features = preprocessing.scale(X_features, with_mean=False, with_std=True, axis=0)

        #from sklearn.model_selection import train_test_split
        #X_train, X_test, Y_train, Y_test = train_test_split(X_features, Y_labels, test_size = validation, random_state = 100)
        #from sklearn.metrics import accuracy_score

        #classifiers['SVR2'].fit(X_train, Y_train)
        for index, (name, classifier) in enumerate(self.classifiers.items()):
            self.classifiers[name].fit(X_features, Y_labels)
            #scores = cross_val_score(classifier, X_train, Y_train_label_ovr, cv=5)
            #y_pred = classifier.predict(X_test)
            #test_acc = accuracy_score(Y_test, y_pred) # same as WAR
            

        return self.classifiers




'''
Feature selection
'''

@jit(nopython=True) 
def get_top_positions(array_y, n_positions):
    order = array_y.argsort()
    ranks = order.argsort() #ascending
    top_indexes = np.zeros((n_positions,), dtype=np.int16)
    #print(array_y)
    i = int(n_positions - 1)

    while(i >= 0):
        itemindices = np.where(ranks==(len(array_y)-1-i))
        for itemindex in itemindices:
            if(itemindex.size):
                #print(i, array_y[itemindex], itemindex)
                top_indexes[i] = itemindex[0]
            else:   #for when positions are more than array size
                itemindices2 = np.where(ranks==len(array_y)-1-i+len(array_y) )
                for itemindex2 in itemindices2:
                    #print(i, array_y[itemindex2], itemindex2)
                    top_indexes[i] = itemindex2[0]
            i -= 1

    return top_indexes
    


def select_features(train_features, Y_labels_train, K_SD=-0.5, mode=0):
    features_n = train_features.shape[1]
    label_classes = np.unique(Y_labels_train)

    if (mode==1):
        #select same number of features optimized for each emotion.
        select_n = int(features_n*np.abs(1-K_SD)/len(label_classes))
        selected_indices = np.zeros((select_n*len(label_classes), ), dtype=np.uint16)

        for e in range(0, len(label_classes)):
            select_emo = np.where(Y_labels_train==label_classes[e])

            features_std_within_emo= np.std(train_features[select_emo], axis=0)
            selected_indices[select_n*e:select_n*(e+1)] = get_top_positions(features_std_within_emo, select_n)
        return selected_indices
    
    elif (mode==2):

        selected_indices = np.zeros(([]), dtype=np.uint16)

        from sklearn.model_selection import ShuffleSplit
        splits_n = 4
        rs = ShuffleSplit(n_splits=splits_n, test_size=None, random_state=0)
        
        feature_means = np.zeros((len(label_classes), splits_n, features_n,), dtype=np.float)
        feature_means_per_emo = np.zeros((len(label_classes), features_n,), dtype=np.float)
        feature_std_per_emo = np.zeros((len(label_classes), features_n,), dtype=np.float)
        #parts_emo_mean_std = np.zeros((splits_n, features_n,), dtype=np.float)
        tsp = 0
        for big_part, s_part in rs.split(train_features):
            for e in range(0, len(label_classes)):
                # select part --> select emotion --> features mean
                train_features_part = train_features[s_part]
                Y_labels_train_part = Y_labels_train[s_part]
                select_emo = np.where(Y_labels_train_part==label_classes[e])
                feature_means[e, tsp, :] = np.mean(train_features_part[select_emo], axis=0)
            # part --> features std across emotions
            tsp += 1
        
        for e in range(0, len(label_classes)):
            feature_means_per_emo[e] = np.mean(feature_means[e], axis=0)
            feature_std_per_emo[e] = np.std(feature_means[e], axis=0)
        
        for e in range(0, len(label_classes)):

            features_mean_values = np.mean(feature_means_per_emo, axis=0)
            emo_mean_diff = np.abs(features_mean_values - feature_means_per_emo[e])

            K = K_SD
            thresh_mean = np.mean(emo_mean_diff) + (np.std(feature_std_per_emo[e])*(K))
            thresh_std = np.mean(feature_std_per_emo[e]) + (np.std(feature_std_per_emo[e])*(1 - K))

            conditions = (emo_mean_diff > thresh_mean) & (feature_std_per_emo[e] < thresh_std)

            #selected_indices = np.where(conditions)[0]
            selected_indices = np.append(selected_indices, np.where(conditions)[0])
            #print(np.where(conditions)[0].size)

        return np.unique(selected_indices)

    else:
        #select features with high SD across emotions
        feature_means = np.zeros((len(label_classes), features_n, ), dtype=np.float)
        #features_std = np.zeros((len(label_classes), features_n, ), dtype=np.float)
        for e in range(0, len(label_classes)):
            select_emo = np.where(Y_labels_train==label_classes[e])
            feature_means[e, :] = np.mean(train_features[select_emo], axis=0)
            #features_std[e, :] = np.std(train_features[select_emo], axis=0)

        emo_mean_std = np.std(feature_means, axis=0)
        
        #selected_indices = get_top_positions(emo_std_mean, selected_features_n)
        thresh = np.mean(emo_mean_std) + (np.std(emo_mean_std) * K_SD)
        selected_indices = np.where(emo_mean_std > thresh)[0]
    
        if (len(selected_indices) < 1):
            return range(0,features_n)
            
        return selected_indices      




'''
Phoneme clustering
'''

def make_clusters(cluster_n1, frames_formants, clustering_method=0, print_quality_scores=0, save_plot=0):

    
    #duration = timer() - start
    #print('Time: %0.1fs' % duration)
    start = timer()

    max_frames = frames_formants.shape[1]
    X_Clips_N = frames_formants.shape[0]
    combined_frames = np.zeros((X_Clips_N*max_frames, frames_formants.shape[2]), dtype=np.uint16)

    #print("Clustering data shape:", combined_frames.shape)

    for i in range(0, X_Clips_N):
        for k in range(0, max_frames):
            combined_frames[(i*max_frames) + k, :] = frames_formants[i, k,:]
    
    clustModel = None

     
    #from sklearn.cluster import Birch
    #clustModel = Birch(n_clusters=None, threshold=0.5, branching_factor=500)
    #clustModel.fit(combined_frames)

    metric='euclidean'
    if(clustering_method==0):
        from sklearn.cluster import MiniBatchKMeans
        clustModel = MiniBatchKMeans(cluster_n1, init='k-means++', batch_size=2000, max_iter=500, n_init=5, compute_labels=False)
        clustModel.fit(combined_frames)
        clustModel.fit(combined_frames)

    elif(clustering_method==1):
        from sklearn.cluster import MiniBatchKMeans
        clustModel = MiniBatchKMeans(cluster_n1, init='k-means++', batch_size=1000, max_iter=500, n_init=5)
        clustModel.fit(combined_frames)
        clustModel.fit(combined_frames)
    
    elif(clustering_method==2):
        from sklearn.cluster import MiniBatchKMeans
        clustModel = MiniBatchKMeans(cluster_n1, init='k-means++', batch_size=2000, max_iter=500, n_init=10)
        clustModel.fit(combined_frames)
        clustModel.fit(combined_frames)
    
    elif(clustering_method==3):
        from sklearn.cluster import MiniBatchKMeans
        clustModel = MiniBatchKMeans(cluster_n1, init='random', batch_size=1000, max_iter=500, n_init=10)
        clustModel.fit(combined_frames)
        clustModel.fit(combined_frames)
        
    elif(clustering_method==4):
        from sklearn.cluster import KMeans
        clustModel = KMeans(cluster_n1, init='k-means++', max_iter=500)
        clustModel.fit(combined_frames)
        
    elif(clustering_method==5):
        from sklearn.cluster import AgglomerativeClustering
        clustModel = AgglomerativeClustering(n_clusters=cluster_n1, linkage="ward", affinity=metric) # only "euclidean" with ward is accepted
        clustModel.fit(combined_frames)

    elif(clustering_method==6):
        from sklearn.cluster import AgglomerativeClustering
        clustModel = AgglomerativeClustering(n_clusters=cluster_n1, linkage="average", affinity=metric) # “euclidean”, “l1”, “l2”, “manhattan”, “cosine”, or “precomputed”
        clustModel.fit(combined_frames)

    elif(clustering_method==7):
        metric='l1'
        from sklearn.cluster import AgglomerativeClustering
        clustModel = AgglomerativeClustering(n_clusters=cluster_n1, linkage="average", affinity=metric) # “euclidean”, “l1”, “l2”, “manhattan”, “cosine”, or “precomputed”
        clustModel.fit(combined_frames)
    
    elif(clustering_method==8):
        metric='l2'
        from sklearn.cluster import AgglomerativeClustering
        clustModel = AgglomerativeClustering(n_clusters=cluster_n1, linkage="average", affinity=metric) # “euclidean”, “l1”, “l2”, “manhattan”, “cosine”, or “precomputed”
        clustModel.fit(combined_frames)
    
    elif(clustering_method==9):
        metric='manhattan'
        from sklearn.cluster import AgglomerativeClustering
        clustModel = AgglomerativeClustering(n_clusters=cluster_n1, linkage="average", affinity=metric) # “euclidean”, “l1”, “l2”, “manhattan”, “cosine”, or “precomputed”
        clustModel.fit(combined_frames)



    

    

    if(print_quality_scores):
        print("Calculating clustering quality scores")
        duration = timer() - start
        
        unique_phonemes = len(np.unique(clustModel.labels_))

        if(unique_phonemes < 2):
            print("Error: Only 1 cluster is created. Please try different clustering parameters.")

        else:
            from sklearn import metrics
            silh_s = metrics.silhouette_score(combined_frames, clustModel.labels_, metric=metric)
            ch_s = metrics.calinski_harabasz_score(combined_frames, clustModel.labels_)
            db_s = metrics.davies_bouldin_score(combined_frames, clustModel.labels_)
            #See details at:
            #https://scikit-learn.org/stable/modules/clustering.html#silhouette-coefficient

            # higher Silhouette Coefficient score relates to a model with better defined clusters
            # Calinski-Harabasz The score is higher when clusters are dense and well separated, which relates to a standard concept of a cluster.
            # Davies-Bouldin Index: Zero is the lowest possible score. Values closer to zero indicate a better partition.
            print('Clustering, Time, Unique Cluster Lables, Silhouette, Calinski-Harabasz, Davies-Bouldin')
            print('%d, %0.1fs,  %d, %0.3f, %d, %0.3f' % (clustering_method, duration, unique_phonemes, silh_s, ch_s, db_s))

            

    if(save_plot):
        print("Creating cluster scatter plot")

        import matplotlib.pyplot as plt

        fig = plt.figure(figsize=(12, 10))
        fig.subplots_adjust(left=0.08, right=0.98, bottom=0.09, top=0.99)
        ax1 = fig.add_subplot(2, 2, 1)
        ax2 = fig.add_subplot(2, 2, 2)
        ax3 = fig.add_subplot(2, 2, 3)
        ax4 = fig.add_subplot(2, 2, 4)

        ax1.set_xlim([600, 1700])
        ax2.set_xlim([600, 1700])
        ax3.set_xlim([600, 1700])
        ax4.set_xlim([600, 1700])
        
        ax1.scatter(combined_frames[:, 0], combined_frames[:, 1], c=clustModel.labels_, cmap=plt.cm.nipy_spectral, alpha=0.25)
        ax1.set_xlabel('f\u2080', fontsize=12, fontstyle='italic', fontfamily='serif')
        ax1.set_ylabel('p\u2080', fontsize=12, fontstyle='italic', fontfamily='serif')

        ax2.scatter(combined_frames[:, 0], combined_frames[:, 4], c=clustModel.labels_, cmap=plt.cm.nipy_spectral, alpha=0.25)
        ax2.set_xlabel('f\u2080', fontsize=12, fontstyle='italic', fontfamily='serif')
        ax2.set_ylabel('f\u2081', fontsize=12, fontstyle='italic', fontfamily='serif')

        ax3.scatter(combined_frames[:, 0], combined_frames[:, 2], c=clustModel.labels_, cmap=plt.cm.nipy_spectral, alpha=0.25)
        ax3.set_xlabel('f\u2080', fontsize=12, fontstyle='italic', fontfamily='serif')
        ax3.set_ylabel('w\u2080', fontsize=12, fontstyle='italic', fontfamily='serif')

        ax4.scatter(combined_frames[:, 0], combined_frames[:, 3], c=clustModel.labels_, cmap=plt.cm.nipy_spectral, alpha=0.25)
        ax4.set_xlabel('f\u2080', fontsize=12, fontstyle='italic', fontfamily='serif')
        ax4.set_ylabel('d\u2080', fontsize=12, fontstyle='italic', fontfamily='serif')

        plt.savefig("clusters_" + str(cluster_n1) + "_" + str(clustering_method) + ".png", dpi=300)
        #plt.show()
        
        
    
    return clustModel







@jit(nopython=True)
def formants_slope(frames_formants, frames_len, distance):
    max_clips = frames_formants.shape[0]
    max_frames = frames_formants.shape[1]
    #max_formants = 2
    slope_ftr = np.zeros((max_clips, max_frames, 6,))
    half_dist = int(distance/2)
    
    for i in range(0, max_clips):
        for frm in range(0, frames_len[i] - distance):
            slope_ftr[i, frm, 0] = np.mean(frames_formants[i, frm:frm+distance, 0])
            slope_ftr[i, frm, 1] = np.mean(frames_formants[i, frm:frm+distance, 1])
            #slope_ftr[i, frm, 2] = np.mean(frames_formants[i, frm:frm+distance, 4])
            
            slope_ftr[i, frm, 2] = np.abs(np.mean(frames_formants[i, frm:frm+half_dist, 0]) - np.mean(frames_formants[i, frm+half_dist:frm+distance, 0]))
            slope_ftr[i, frm, 3] = np.abs(np.mean(frames_formants[i, frm:frm+half_dist, 1]) - np.mean(frames_formants[i, frm+half_dist:frm+distance, 1]))
            #slope_ftr[i, frm, 5] = np.abs(np.mean(frames_formants[i, frm:frm+half_dist, 4]) - np.mean(frames_formants[i, frm+half_dist:frm+distance, 4]))

    return slope_ftr


@jit(nopython=True)
def count_phonemes_in_clip(phonemes_array, frames_len, phoneme_types):
    phoneme_count = np.zeros((phoneme_types,))
    
    for frm in range(0, frames_len):
        phoneme_count[phonemes_array[frm]] += 1000
    
    phoneme_count = phoneme_count/(frames_len+1)
    return phoneme_count


def count_phonemes_in_set(clustModel, frames_formants, frames_len,  phoneme_types):
    max_clips = frames_formants.shape[0]
    clips_phoneme_count = np.zeros((max_clips, phoneme_types,), dtype=np.float)

    for i in range(0, max_clips):
        phonemes_array = clustModel.predict(frames_formants[i])
        clips_phoneme_count[i] = count_phonemes_in_clip(phonemes_array, frames_len[i], phoneme_types)

    return clips_phoneme_count






def Train_model(models_save_file, X_formants_train, Y_labels_train, X_frame_lens_train, inst_phoneme_types=[16,32,64], diff_phoneme_types=[16,32,64], K_SD=0, g_dist=8):
    '''
    Create clustering models, select phoneme features, train classifiers and save everything to a model file so that it can used later for testing.

    Parameters
    ----------

    `models_save_file`: string, file path to which clustering+classifying models are saved;

    `X_formants_train`: array-like, shape = [n_clips, max_frames, n_formant_features]. Formant features for N utterences with fixed number of frames. Pass the actual frames length of each clip as array 'X_frame_lens_train' with the same order.

    `Y_labels_train`: int array, shape = [n_clips]. Labels all clips in X_formants_train in the same order.

    `X_frame_lens_train`: int array, shape = [n_clips]. Array of number of filled frames of each clip out of max_frames (max_frames default=800).
    
    `inst_phoneme_types`: array-like, dtype=int16, shape = [n_models], optional (default=[16, 32, 64]). Cluster numbers for instantaneous (~25ms) phoneme clustering model. Set between 8 to 300 for each cluster model.;
    
    `diff_phoneme_types`:  array-like, dtype=int16, shape = [n_models], optional (default=[16, 32, 64]). Cluster numbers for differential phoneme (~25ms * g_dist) clustering models. Set between 8 to 300 for each cluster model.;

    `K_SD`: float, optional (default=0.0). Feature selection parameters. Set between -1 to 1. It sets the limit of standard deviation below the mean for selecting features within this threshold. Lower value selects more features;

    `g_dist`: unsigned int, optional (default=8). Number for adjacent frames for measuring the change in formant features to calculate differential phoneme features;
    
    Returns
    -------
    `Trained_classifiers`= `ClassifierObject()`, includes multiple trained classifiers, it doesn't need to be passed, function 'Test_model' reads this automatically from `models_save_file`
    '''

    print("Clustering (inst.)", inst_phoneme_types)
    inst_clust_models = []
    for pt in range(0, len(inst_phoneme_types)):
        #print("Clustering phonemes (inst.) using training clips, Type:", inst_phoneme_types[pt])

        #for clm in range(7):
        pt_clustModel = make_clusters(inst_phoneme_types[pt], X_formants_train, clustering_method=0, print_quality_scores=0, save_plot=0)
        
        #exit()

        inst_clust_models.append(pt_clustModel)


    train_slope_ftrs = formants_slope(X_formants_train, X_frame_lens_train, distance=g_dist)

    print("Clustering (diff.)", diff_phoneme_types)
    diff_clust_models = []
    for pt in range(0, len(diff_phoneme_types)):
        #print("Clustering phonemes (diff.) using training clips, Type:", diff_phoneme_types[pt])
        slope_clustModel = make_clusters(diff_phoneme_types[pt], train_slope_ftrs, clustering_method=0, print_quality_scores=0, save_plot=0)
        
        diff_clust_models.append(slope_clustModel)

    
    max_clips_train = X_formants_train.shape[0]

    print("Counting phonemes")
    all_ftrs = int(np.sum(inst_phoneme_types) +  np.sum(diff_phoneme_types) + 1) #+ np.sum(comb_phoneme_type**2))
    trainclips_phoneme_count = np.zeros((max_clips_train, all_ftrs,), dtype=np.float)

    #print("Counting phonemes (single). Types:", inst_phoneme_types)
    for pt in range(0, len(inst_phoneme_types)):
        stx = int(np.sum(inst_phoneme_types[0:pt]))
        edx = stx + inst_phoneme_types[pt]
        trainclips_phoneme_count[:, stx:edx] = count_phonemes_in_set(inst_clust_models[pt], X_formants_train, X_frame_lens_train, inst_phoneme_types[pt])


    #print("Counting phonemes (slopes). Types:", diff_phoneme_types)
    for pt in range(0, len(diff_phoneme_types)):
        stx = int(np.sum(inst_phoneme_types) + np.sum(diff_phoneme_types[0:pt]))
        edx = stx + diff_phoneme_types[pt]
        trainclips_phoneme_count[:, stx:edx] = count_phonemes_in_set(diff_clust_models[pt], train_slope_ftrs, X_frame_lens_train, diff_phoneme_types[pt])


    trainclips_phoneme_count[:, all_ftrs - 1] = X_frame_lens_train

    selected_features = select_features(trainclips_phoneme_count, Y_labels_train, K_SD=K_SD, mode=0)

    ClassifierObj = ClassifierObject()
    print("Training Classifiers", "(samples, features):", trainclips_phoneme_count[:, selected_features].shape)
    Trained_classifiers = ClassifierObj.Train(trainclips_phoneme_count[:, selected_features], Y_labels_train)
    

    #Save trained models to pickle file
    import pickle
    tuple_objects = (inst_phoneme_types, diff_phoneme_types, g_dist, inst_clust_models, diff_clust_models, selected_features, Trained_classifiers)
    pickle.dump(tuple_objects, open(models_save_file, 'wb'))

    #print("Training complete")

    return Trained_classifiers


def Test_model(models_save_file, X_formants_test, Y_labels_test, X_frame_lens_test):

    '''
    Predict emotion classes using array-like features input of n_clips

    Parameters
    ----------

    `models_save_file`: string, file path where clustering+classifying models are saved, use `Train_model()` to create a model file.

    `X_formants_test`: array-like, shape = [n_clips, max_frames, n_formant_features]. Formant features for N utterences with fixed number of frames. Pass the actual frames length of each clip as array 'X_frame_lens_test' with the same order.

    `Y_labels_test`: int array, shape = [n_clips]. Labels all clips in X_formants_test in the same order.

    `X_frame_lens_test`: int array, shape = [n_clips]. Array of number of filled frames of each clip out of max_frames.
    
    Returns
    -------

    `classifiers_results`: Dict list, shape: [{'classifier' : <classifier_name, string>, 'confusion' : <conf_matrix, array-like>, 'UAR' : <uar, float>, 'WAR' : <war, float>}]

    `n_selected_features`: int, number of selected features.
    '''

    #Load clusering and ML models from pickle file
    import pickle
    inst_phoneme_types, diff_phoneme_types, g_dist, inst_clust_models, diff_clust_models, selected_features, Trained_classifiers = pickle.load(open(models_save_file, 'rb'))

    test_slope_ftrs = formants_slope(X_formants_test, X_frame_lens_test, distance=g_dist)
    max_clips_test = X_formants_test.shape[0]

    all_ftrs = int(np.sum(inst_phoneme_types) +  np.sum(diff_phoneme_types) + 1) #+ np.sum(comb_phoneme_type**2))
    testclips_phoneme_count = np.zeros((max_clips_test, all_ftrs,), dtype=np.float)

    #print("Counting phonemes (single). Types:", inst_phoneme_types)
    for pt in range(0, len(inst_phoneme_types)):
        stx = int(np.sum(inst_phoneme_types[0:pt]))
        edx = stx + inst_phoneme_types[pt]
        testclips_phoneme_count[:, stx:edx] = count_phonemes_in_set(inst_clust_models[pt], X_formants_test, X_frame_lens_test, inst_phoneme_types[pt])
    

    #print("Counting phonemes (slopes). Types:", diff_phoneme_types)
    for pt in range(0, len(diff_phoneme_types)):
        stx = int(np.sum(inst_phoneme_types) + np.sum(diff_phoneme_types[0:pt]))
        edx = stx + diff_phoneme_types[pt]
        testclips_phoneme_count[:, stx:edx] = count_phonemes_in_set(diff_clust_models[pt], test_slope_ftrs, X_frame_lens_test, diff_phoneme_types[pt])


    testclips_phoneme_count[:, all_ftrs - 1] = X_frame_lens_test

    classifiers_results = []

    for index, (name, classifier) in enumerate(Trained_classifiers.items()):
        y_pred = Test(classifier, testclips_phoneme_count[:, selected_features])
        conf_matrix = create_conf_matrix(Y_labels_test, y_pred, np.unique(Y_labels_test))
        uar, war = get_UAR_WAR(conf_matrix)
        #print(name, uar, war)
        classifiers_results.append({'classifier' : name, 'confusion' : conf_matrix, 'UAR' : uar, 'WAR' : war})
    
    ##################
    #print([chr(x) for x in #label_classes])

    n_selected_features = len(selected_features)
    
    return classifiers_results, n_selected_features


def Test_model_wav_file(models_save_file, test_file):
    '''
    Predict emotion class of a WAV file

    Parameters
    ----------

    `models_save_file`: string, file path where clustering+classifying models are saved, use `Train_model()` to create a model file.

    `test_file`: string, path to the WAV file.

    Returns
    -------

    `classifiers_results`: Dict list of predicted classes by all classifiers, shape = [{'classifier' : <classifier_name>, 'Prediction' : '<label>'}]

    '''

    from FormantsLib.FormantsExtract import Extract_wav_file_formants

    array_frames_by_features, frame_count, signal_length, trimmed_length = Extract_wav_file_formants(test_file, window_length=0.025, window_step=0.01, emphasize_ratio=0.65, f0_min=30, f0_max=4000, max_frames=800, formants=3)

    
    X_formants_test = np.array([array_frames_by_features])
    
    X_frame_lens_test = np.array([frame_count])
    #print("Testing")
    #Load clusering and ML models from pickle file
    import pickle
    inst_phoneme_types, diff_phoneme_types, g_dist, inst_clust_models, diff_clust_models, selected_features, Trained_classifiers = pickle.load(open(models_save_file, 'rb'))

    test_slope_ftrs = formants_slope(X_formants_test, X_frame_lens_test, distance=g_dist)
    max_clips_test = X_formants_test.shape[0]
    
    all_ftrs = int(np.sum(inst_phoneme_types) +  np.sum(diff_phoneme_types) + 1) #+ np.sum(comb_phoneme_type**2))
    testclips_phoneme_count = np.zeros((max_clips_test, all_ftrs,), dtype=np.float)

    #print("Counting phonemes (single). Types:", inst_phoneme_types)
    for pt in range(0, len(inst_phoneme_types)):
        stx = int(np.sum(inst_phoneme_types[0:pt]))
        edx = stx + inst_phoneme_types[pt]
        testclips_phoneme_count[:, stx:edx] = count_phonemes_in_set(inst_clust_models[pt], X_formants_test, X_frame_lens_test, inst_phoneme_types[pt])
    

    #print("Counting phonemes (slopes). Types:", diff_phoneme_types)
    for pt in range(0, len(diff_phoneme_types)):
        stx = int(np.sum(inst_phoneme_types) + np.sum(diff_phoneme_types[0:pt]))
        edx = stx + diff_phoneme_types[pt]
        testclips_phoneme_count[:, stx:edx] = count_phonemes_in_set(diff_clust_models[pt], test_slope_ftrs, X_frame_lens_test, diff_phoneme_types[pt])


    testclips_phoneme_count[:, all_ftrs - 1] = X_frame_lens_test

    classifiers_results = []

    for index, (name, classifier) in enumerate(Trained_classifiers.items()):
        y_pred = Test(classifier, testclips_phoneme_count[:, selected_features])
        
        print(name, [chr(x) for x in y_pred][0])
        classifiers_results.append({'classifier' : name, 'Prediction' : [chr(x) for x in y_pred][0]})
    
    return classifiers_results


