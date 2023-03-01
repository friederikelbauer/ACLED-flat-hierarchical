# basics
from os import cpu_count

# model
from src.models.RF import *
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

# getting the data
train = pd.read_pickle("data/processed/long_train.pkl")
test = pd.read_pickle("data/processed/long_test.pkl")

train, test = add_third_level(train, test)

# splitting in train and test
x_train = train.notes
y_train = train[["first_level", "event_type", "sub_event_type"]]
x_test = test.notes
y_test = test[["first_level", "event_type", "sub_event_type"]]

# model
classifier = createRF().get_classifier()
hierachical_classifier = LocalClassifierPerParentNode(
    local_classifier=classifier, n_jobs=cpu_count()
)

########
# COUNT VECTORIZER
########

# # Count Vectorizer
X_train, X_test = createCountVectorizer().transform(x_train, x_test)

# Hierarchical rf
hierachical_classifier.fit(X_train, y_train)
y_pred = hierachical_classifier.predict(X_test)
y_pred = [list(map(str, lst)) for lst in y_pred]

# Evaluation
evaluation = ModelEvaluator(
    y_test, y_pred, name="rf/three_levels_rf_hierarchical_count_vectorizer"
)

# second level
evaluation.evaluate_hierarchical(
    make_matrix=False,
    hierarchy_level=2,
)
evaluation.save_hierarchy_predictions(x_test=x_test)

########
# TF-IDF
########

# rf
X_train, X_test = createTFIDF().transform(x_train, x_test)

# Hierarchical rf
hierachical_classifier.fit(X_train, y_train)
y_pred = hierachical_classifier.predict(X_test)
y_pred = [list(map(str, lst)) for lst in y_pred]

# Evaluation
evaluation = ModelEvaluator(
    y_test, y_pred, name="rf/three_levels_rf_hierarchical_tfidf"
)
# first level

# second level
evaluation.evaluate_hierarchical(
    make_matrix=False,
    hierarchy_level=2,
)
evaluation.save_hierarchy_predictions(x_test=x_test)

########
# FastText
########

# FastText
X_train, X_test = createFastText().transform(x_train, x_test)

# Hierarchical rf
hierachical_classifier.fit(X_train, y_train)
y_pred = hierachical_classifier.predict(X_test)
y_pred = [list(map(str, lst)) for lst in y_pred]

# # Evaluation
evaluation = ModelEvaluator(
    y_test, y_pred, name="rf/three_levels_rf_hierarchical_fasttext"
)

# second level
evaluation.evaluate_hierarchical(
    make_matrix=False,
    hierarchy_level=2,
)
evaluation.save_hierarchy_predictions(x_test=x_test)

########
# Word2Vec
########

# Word2Vec
X_train, X_test = createW2V().transform(x_train, x_test)

# Hierarchical Word2Vec
hierachical_classifier.fit(X_train, y_train)
y_pred = hierachical_classifier.predict(X_test)
y_pred = [list(map(str, lst)) for lst in y_pred]


# # Evaluation
evaluation = ModelEvaluator(
    y_test, y_pred, name="rf/three_levels_rf_hierarchical_word2vec"
)

# second level
evaluation.evaluate_hierarchical(
    make_matrix=False,
    hierarchy_level=2,
)
evaluation.save_hierarchy_predictions(x_test=x_test)


#####
# BERT Vectorizer
#####

# BERT Vectorizer
X_train, X_test = createBERTVectorizer().transform(x_train, x_test)

# Hierarchical rf
hierachical_classifier.fit(X_train, y_train)
y_pred = hierachical_classifier.predict(X_test)
y_pred = [list(map(str, lst)) for lst in y_pred]

# Evaluation
evaluation = ModelEvaluator(
    y_test, y_pred, name="rf/three_levels_rf_hierarchical_bert_vectorizer"
)

# second level
evaluation.evaluate_hierarchical(
    make_matrix=False,
    hierarchy_level=2,
)
evaluation.save_hierarchy_predictions(x_test=x_test)
