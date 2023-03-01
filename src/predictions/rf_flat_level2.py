# basics
import logging
from datetime import datetime

# model
from src.models.RF import *

# Vectorizers
from src.feature_extractors.CountVectorizer import *
from src.feature_extractors.TFIDF import *
from src.feature_extractors.Word2Vec import *
from src.feature_extractors.FastText import *
from src.feature_extractors.BERT_Vectorizer import *

# evaluation
from src.utils.evaluation_functions import *
from src.utils.cross_validation import *


logging.basicConfig(
    filename="src/logs/rf_flat_level2.log", level=logging.DEBUG, filemode="w"
)
logging.getLogger("matplotlib.font_manager").disabled = True
logging.getLogger("matplotlib.pyplot").disabled = True
logging.getLogger("matplotlib.colorbar").disabled = True

# getting the data
train = pd.read_pickle("data/processed/long_train.pkl")
test = pd.read_pickle("data/processed/long_test.pkl")

# splitting in train and test
x_train = train.notes
y_train = train.sub_event_type
x_test = test.notes
y_test = test.sub_event_type


########
# COUNT VECTORIZER
########
start = datetime.now()
logging.info(f"starting count vectorizer at {start}")

# Count Vectorizer
X_train, X_test = createCountVectorizer().transform(x_train, x_test)

# NaiveBayes
y_pred = createRF().fit_classifier(X_train, y_train, X_test)

end = datetime.now()
logging.info(f"ending count vectorizer at {end}, took {end-start}")

# Evaluation
evaluation = ModelEvaluator(y_test, y_pred, name="rf/rf_flat_count_vectorizer_level2")
evaluation.evaluate_flat(level=2)
evaluation.make_confusionmatrix()
evaluation.save_flat_predictions(x_test=x_test)

# Cross Validation
# flat_cross_validate(
#     local_classifier=createRF(),
#     vectorizer=createCountVectorizer(),
#     X=x_train,
#     y=y_train,
#     name="rf/flat_crossvalidation_rf_countvectorizer_level2",
# )


########
# TF-IDF
########
start = datetime.now()
logging.info(f"starting TF-IDF at {start}")

# TF-IDF
X_train, X_test = createTFIDF().transform(x_train, x_test)

# NaiveBayes
y_pred = createRF().fit_classifier(X_train, y_train, X_test)

end = datetime.now()
logging.info(f"ending TF-IDF at {end}, took {end-start}")

# Evaluation
evaluation = ModelEvaluator(y_test, y_pred, name="rf/rf_flat_tfidf_level2")
evaluation.evaluate_flat(level=2)
evaluation.make_confusionmatrix()
evaluation.save_flat_predictions(x_test=x_test)


# Cross Validation
# flat_cross_validate(
#     local_classifier=createRF(),
#     vectorizer=createTFIDF(),
#     X=x_train,
#     y=y_train,
#     name="rf/flat_crossvalidation_rf_tfidf_level2",
# )

########
# FastText
########
start = datetime.now()
logging.info(f"starting FastText at {start}")

# FastText
X_train, X_test = createFastText().transform(x_train, x_test)

# LogisticRegression
y_pred = createRF().fit_classifier(X_train, y_train, X_test)

end = datetime.now()
logging.info(f"ending FastText at {end}, took {end-start}")

# Evaluation
evaluation = ModelEvaluator(y_test, y_pred, name="rf/rf_flat_fasttext_level2")
evaluation.evaluate_flat(level=2)
evaluation.make_confusionmatrix()
evaluation.save_flat_predictions(x_test=x_test)


# flat_cross_validate(
#     local_classifier=createRF(),
#     vectorizer=createFastText(),
#     X=x_train,
#     y=y_train,
#     name="rf/flat_crossvalidation_rf_fasttext_level2",
# )


########
# Word2Vec
########
start = datetime.now()
logging.info(f"starting Word2Vec at {start}")

# Word2Vec
X_train, X_test = createW2V().transform(x_train, x_test)

# NaiveBayes
y_pred = createRF().fit_classifier(X_train, y_train, X_test)

end = datetime.now()
logging.info(f"ending Word2Vec at {end}, took {end-start}")

# Evaluation
evaluation = ModelEvaluator(y_test, y_pred, name="rf/rf_flat_word2vec_level2")
evaluation.evaluate_flat(level=2)
evaluation.make_confusionmatrix()
evaluation.save_flat_predictions(x_test=x_test)


# Cross Validation
# flat_cross_validate(
#     local_classifier=createRF(),
#     vectorizer=createW2V(),
#     X=x_train,
#     y=y_train,
#     name="rf/flat_crossvalidation_rf_word2vec_level2",
# )

########
# BERT Vectorizer
########

start = datetime.now()
logging.info(f"starting BERT Vectorizer at {start}")

# # Bert Vectorizer
X_train, X_test = createBERTVectorizer().transform(x_train, x_test)

# Random Forest
y_pred = createRF().fit_classifier(X_train, y_train, X_test)

end = datetime.now()
logging.info(f"ending BERT Vectorizer at {end}, took {end-start}")

# # Evaluation
evaluation = ModelEvaluator(
    y_test,
    y_pred,
    name="rf/rf_flat_bert_vectorizer_level2",
)
evaluation.evaluate_flat(level=2)
evaluation.save_flat_predictions(x_test=x_test)
evaluation.make_confusionmatrix()

# Cross Validation
# flat_cross_validate(
#     local_classifier=createRF(),
#     vectorizer=createBERTVectorizer(),
#     X=x_train,
#     y=y_train,
#     name="rf/flat_crossvalidation_rf_bert_vectorizer_level2",
# )
