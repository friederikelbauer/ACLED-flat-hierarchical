# basics
import logging
from datetime import datetime
from os import cpu_count

# model
from src.models.LR import *
from hiclass import LocalClassifierPerParentNode

# Vectorizers
from src.feature_extractors.CountVectorizer import *
from src.feature_extractors.TFIDF import *
from src.feature_extractors.Word2Vec import *
from src.feature_extractors.FastText import *
from src.feature_extractors.BERT_Vectorizer import *

# evaluation
from src.utils.evaluation_functions import *
from src.utils.cross_validation import *

# logger configuration
logging.basicConfig(
    filename="src/logs/lr_hierarchical.log", level=logging.DEBUG, filemode="w"
)
logging.getLogger("matplotlib.font_manager").disabled = True
logging.getLogger("matplotlib.pyplot").disabled = True
logging.getLogger("matplotlib.colorbar").disabled = True

# getting the data
train = pd.read_pickle("data/processed/long_train.pkl")
test = pd.read_pickle("data/processed/long_test.pkl")

# splitting in train and test
x_train = train.notes
y_train = train[["event_type", "sub_event_type"]]
x_test = test.notes
y_test = test[["event_type", "sub_event_type"]]

# model
classifier = createLogisticRegression().get_classifier()
hierachical_classifier = LocalClassifierPerParentNode(
    local_classifier=classifier, n_jobs=cpu_count()
)

######
# COUNT VECTORIZER
######
start = datetime.now()
logging.info(f"starting count vectorizer at {start}")

# Count Vectorizer
X_train, X_test = createCountVectorizer().transform(x_train, x_test)

# Hierarchical Logistic Regression
hierachical_classifier.fit(X_train, y_train)
y_pred = hierachical_classifier.predict(X_test)
y_pred = [list(map(str, lst)) for lst in y_pred]


end = datetime.now()
logging.info(f"ending count vectorizer at {end}, took {end-start}")

# checking consistency
checking_consistency(y_pred)

# Evaluation
evaluation = ModelEvaluator(y_test, y_pred, name="lr/lr_hierarchical_count_vectorizer")
# first level
evaluation.evaluate_hierarchical(
    make_matrix=False,
    hierarchy_level=1,
)
# second level
evaluation.evaluate_hierarchical(
    make_matrix=False,
    hierarchy_level=2,
)
evaluation.save_hierarchy_predictions(x_test=x_test)


# Cross Validation
# hierarchical_cross_validate(
#     local_classifier=createLogisticRegression().get_classifier(),
#     vectorizer=createCountVectorizer(),
#     X=x_train,
#     y=y_train,
#     name="lr/hierarchical_crossvalidation_lr_countvectorizer",
# )


# ########
# # TF-IDF
# ########
start = datetime.now()
logging.info(f"starting TF-IDF at {start}")

# TF-IDF
X_train, X_test = createTFIDF().transform(x_train, x_test)

# Hierarchical Logistic Regression
hierachical_classifier.fit(X_train, y_train)
y_pred = hierachical_classifier.predict(X_test)
y_pred = [list(map(str, lst)) for lst in y_pred]

end = datetime.now()
logging.info(f"ending TF-IDF at {end}, took {end-start}")

# checking consistency
checking_consistency(y_pred)

# Evaluation
evaluation = ModelEvaluator(y_test, y_pred, name="lr/lr_hierarchical_tfidf")
# first level
evaluation.evaluate_hierarchical(
    make_matrix=False,
    hierarchy_level=1,
)
# second level
evaluation.evaluate_hierarchical(
    make_matrix=False,
    hierarchy_level=2,
)
evaluation.save_hierarchy_predictions(x_test=x_test)


# Cross Validation
# hierarchical_cross_validate(
#     local_classifier=createLogisticRegression().get_classifier(),
#     vectorizer=createTFIDF(),
#     X=x_train,
#     y=y_train,
#     name="lr/hierarchical_crossvalidation_lr_tfidf",
# )

#######
# FastText
#######
start = datetime.now()
logging.info(f"starting FastText at {start}")

# FastText
X_train, X_test = createFastText().transform(x_train, x_test)

# Hierarchical LogisticRegression
hierachical_classifier.fit(X_train, y_train)
y_pred = hierachical_classifier.predict(X_test)
y_pred = [list(map(str, lst)) for lst in y_pred]

end = datetime.now()
logging.info(f"ending FastText at {end}, took {end-start}")

# checking consistency
checking_consistency(y_pred)

# Evaluation
evaluation = ModelEvaluator(y_test, y_pred, name="lr/lr_hierarchical_fasttext")
# first level
evaluation.evaluate_hierarchical(
    make_matrix=False,
    hierarchy_level=1,
)
# second level
evaluation.evaluate_hierarchical(
    make_matrix=False,
    hierarchy_level=2,
)
evaluation.save_hierarchy_predictions(x_test=x_test)


# # Cross Validation
# hierarchical_cross_validate(
#     local_classifier=createLogisticRegression().get_classifier(),
#     vectorizer=createFastText(),
#     X=x_train,
#     y=y_train,
#     name="lr/hierarchical_crossvalidation_lr_fasttext",
# )

########
# Word2Vec
########
start = datetime.now()
logging.info(f"starting Word2Vec at {start}")

# Word2Vec
X_train, X_test = createW2V().transform(x_train, x_test)

# Hierarchical Logistic Regression
hierachical_classifier.fit(X_train, y_train)
y_pred = hierachical_classifier.predict(X_test)
y_pred = [list(map(str, lst)) for lst in y_pred]

end = datetime.now()
logging.info(f"ending Word2Vec at {end}, took {end-start}")

# checking consistency
checking_consistency(y_pred)

# Evaluation
evaluation = ModelEvaluator(y_test, y_pred, name="lr/lr_hierarchical_word2vec")
# first level
evaluation.evaluate_hierarchical(
    make_matrix=False,
    hierarchy_level=1,
)
# second level
evaluation.evaluate_hierarchical(
    make_matrix=False,
    hierarchy_level=2,
)
evaluation.save_hierarchy_predictions(x_test=x_test)


# Cross Validation
# hierarchical_cross_validate(
#     local_classifier=createLogisticRegression().get_classifier(),
#     vectorizer=createW2V(),
#     X=x_train,
#     y=y_train,
#     name="lr/hierarchical_crossvalidation_lr_word2vec",
# )

# #######
# # BERT Vectorizer
# #######

start = datetime.now()
logging.info(f"starting BERT Vectorizer at {start}")

# BERT Vectorizer
X_train, X_test = createBERTVectorizer().transform(x_train, x_test)

# Hierarchical LogisticRegression
hierachical_classifier.fit(X_train, y_train)
y_pred = hierachical_classifier.predict(X_test)
y_pred = [list(map(str, lst)) for lst in y_pred]

end = datetime.now()
logging.info(f"ending BERT vectorizer at {end}, took {end-start}")

# checking consistency
checking_consistency(y_pred)

# Evaluation
evaluation = ModelEvaluator(y_test, y_pred, name="lr/lr_hierarchical_bert_vectorizer")
# first level
evaluation.evaluate_hierarchical(
    make_matrix=False,
    hierarchy_level=1,
)
# second level
evaluation.evaluate_hierarchical(
    make_matrix=False,
    hierarchy_level=2,
)
evaluation.save_hierarchy_predictions(x_test=x_test)


# Cross Validation
# hierarchical_cross_validate(
#     local_classifier=createLogisticRegression().get_classifier(),
#     vectorizer=createBERTVectorizer(),
#     X=x_train,
#     y=y_train,
#     name="lr/hierarchical_crossvalidation_lr_bert_vectorizer",
# )
