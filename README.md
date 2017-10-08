### iDTI-ESBoost: Identiﬁcation of Drug Target Interaction Using Evolutionary and Structural Features with Boosting

# ABSTRACT
Prediction of new drug-target interactions is extremely important as it can lead the researchers to ﬁnd new uses for old drugs and to realize the therapeutic proﬁles or side effects thereof. However, experimental prediction of drug-target interactions is expensive and time-consuming. As a result, computational methods for prediction of new drug-target interactions have gained much interest in recent times. We present iDTI-ESBoost, a prediction model for identiﬁcation of drug-target interactions using evolutionary and structural features. Our proposed method uses a novel balancing technique and a boosting technique for the binary classiﬁcation problem of drug-target interaction. On four benchmark datasets taken from a gold standard data, iDTI-ESBoost outperforms the state-of-the-art methods in terms of area under Receiver operating characteristic (auROC) curve. iDTI-ESBoost also outperforms the latest and the best-performing method in the literature to-date in terms of area under precision recall (auPR) curve. This is signiﬁcant as auPR curves are argued to be more appropriate as a metric for comparison for imbalanced datasets, like the one studied in this research. In the sequel, our experiments establish the effectiveness of the classiﬁer, balancing methods and the novel features incorporated in iDTI-ESBoost. iDTI-ESBoost is a novel prediction method that has for the ﬁrst time exploited the structural features along with the evolutionary features to predict drug-protein interactions. We believe the excellent performance of iDTI-ESBoost both in terms of auROC and auPR would motivate the researchers and practitioners to use it to predict drug-target interactions. To facilitate that, iDTI-ESBoost is readily available for use at: http://farshidrayhan.pythonanywhere.com/iDTI-ESBoost/.


## Feature Generation codes

PSSM bigram => A   
pssm_to_bigram_generator.py file was used to create the feature set A  

Accessible Surface Area Composition + Secondary Structure Composition + Torsional Angles Composition => B  
creating_structural_info.py file was used to create the feature set B  

Torsional Angles Auto-Covariance + Structural Probabilities Auto-Covariance => C  
spider_to_auto_covar_generator.py file was used to create the feature set C  

Torsional Angles bigram + Structural Probabilities bigram => D   
spider_to_bigram_generator.py file was used to create the feature set D  

## Experiments 

Use the codes named  

cluster_based_under_sampling.py  
and  
random_under_sampling.py    

for experimenting a dataset with a classifier.     
