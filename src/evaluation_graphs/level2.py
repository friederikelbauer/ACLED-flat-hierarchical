import matplotlib.pyplot as plt
import scienceplots

plt.style.use("science")

# data
mode = ["flat", "hierarchical"]


#########
# ACCURACY
#########

# Initialise the subplot function using number of rows and columns
figure, ((axis1, axis4), (axis2, axis5), (axis3, axis6)) = plt.subplots(
    nrows=3, ncols=2
)

# upper left
axis1.plot(mode, [0.9179, 0.9146], label="Count Vectorizer")
axis1.plot(mode, [0.9137, 0.9134], label="TF-IDF")
axis1.plot(mode, [0.8303, 0.8115], label="Word2Vec")
axis1.plot(mode, [0.7557, 0.7419], label="Fast Text")
axis1.plot(mode, [0.8866, 0.8750], label="BERT Vectorizer")
axis1.set_title("Logistic Regression - accuracy")
axis1.set(ylabel="accuracy")
figure.legend(facecolor="white", framealpha=0, loc="center right")

# upper right
axis2.plot(mode, [0.9021, 0.9023])
axis2.plot(mode, [0.9139, 0.9161])
axis2.plot(mode, [0.8120, 0.8040])
axis2.plot(mode, [0.7664, 0.7548])
axis2.plot(mode, [0.8733, 0.8692])
axis2.set_title("Support Vector Machine - accuracy")
axis2.set(ylabel="accuracy")
figure.legend(facecolor="white", framealpha=0, loc="center right")

# lower left
axis3.plot(mode, [0.8736, 0.8767])
axis3.plot(mode, [0.8755, 0.8726])
axis3.plot(mode, [0.7618, 0.7647])
axis3.plot(mode, [0.6632, 0.6688])
axis3.plot(mode, [0.7574, 0.7645])
axis3.set(ylabel="accuracy")
axis3.set_title("Random Forest - accuracy")

figure.legend(facecolor="white", framealpha=0, loc="center right")


#########
# F1 MACRO
#########

# upper left
axis4.plot(mode, [0.8665, 0.8633])
axis4.plot(mode, [0.8442, 0.8443])
axis4.plot(mode, [0.712, 0.6966])
axis4.plot(mode, [0.5670, 0.5525])
axis4.plot(mode, [0.8400, 0.8176])
axis4.set_title("Logistic Regression - f1 macro")
axis4.set(ylabel="f1 macro")
figure.legend(facecolor="white", framealpha=0, loc="center right")

# upper right
axis5.plot(mode, [0.8565, 0.8548])
axis5.plot(mode, [0.8725, 0.8729])
axis5.plot(mode, [0.7113, 0.7010])
axis5.plot(mode, [0.6280, 0.6074])
axis5.plot(mode, [0.8257, 0.8151])
axis5.set_title("Support Vector Machine - f1 macro")
axis5.set(ylabel="f1 macro")
figure.legend(facecolor="white", framealpha=0, loc="center right")

# lower left
axis6.plot(mode, [0.7646, 0.7710])
axis6.plot(mode, [0.7687, 0.7631])
axis6.plot(mode, [0.5669, 0.5925])
axis6.plot(mode, [0.4450, 0.4637])
axis6.plot(mode, [0.5600, 0.5923])
axis6.set(ylabel="f1 macro")
axis6.set_title("Random Forest - f1 macro")

figure.legend(facecolor="white", framealpha=0, loc="center right")
plt.setp((axis1, axis2, axis3, axis4, axis5, axis6), ylim=(0.4, 1))
plt.show()

# #########
# # ACCURACY
# #########

# # Initialise the subplot function using number of rows and columns
# figure, (axis1, axis2, axis3) = plt.subplots(nrows=1, ncols=3)

# # upper left
# axis1.plot(mode, [0.9179, 0.9146], label="Count Vectorizer")
# axis1.plot(mode, [0.9137, 0.9134], label="TF-IDF")
# axis1.plot(mode, [0.8303, 0.8115], label="Word2Vec")
# axis1.plot(mode, [0.7557, 0.7419], label="Fast Text")
# axis1.plot(mode, [0.8866, 0.8750], label="BERT Vectorizer")
# axis1.set_title("Logistic Regression - accuracy")
# axis1.set(ylabel="accuracy")
# figure.legend(facecolor="white", framealpha=0, loc="center right")

# # upper right
# axis2.plot(mode, [0.9021, 0.9023])
# axis2.plot(mode, [0.9139, 0.9161])
# axis2.plot(mode, [0.8120, 0.8040])
# axis2.plot(mode, [0.7664, 0.7548])
# axis2.plot(mode, [0.8733, 0.8692])
# axis2.set_title("Support Vector Machine - accuracy")
# figure.legend(facecolor="white", framealpha=0, loc="center right")

# # lower left
# axis3.plot(mode, [0.8736, 0.8767])
# axis3.plot(mode, [0.8755, 0.8726])
# axis3.plot(mode, [0.7618, 0.7647])
# axis3.plot(mode, [0.6632, 0.6688])
# axis3.plot(mode, [0.7574, 0.7645])
# axis3.set_title("Random Forest - accuracy")

# figure.legend(facecolor="white", framealpha=0, loc="center right")

# # Setting the values for all axes.
# plt.setp((axis1, axis2, axis3), ylim=(0, 1))
# plt.show()


# #########
# # F1 MACRO
# #########

# # Initialise the subplot function using number of rows and columns
# figure, (axis1, axis2, axis3) = plt.subplots(nrows=2, ncols=3)

# # upper left
# axis1.plot(mode, [0.8665, 0.8633], label="Count Vectorizer")
# axis1.plot(mode, [0.8442, 0.8443], label="TF-IDF")
# axis1.plot(mode, [0.712, 0.6966], label="Word2Vec")
# axis1.plot(mode, [0.5670, 0.5525], label="Fast Text")
# axis1.plot(mode, [0.8400, 0.8176], label="BERT Vectorizer")
# axis1.set_title("Logistic Regression - f1 macro")
# axis1.set(ylabel="f1 macro")
# figure.legend(facecolor="white", framealpha=0, loc="center right")

# # upper right
# axis2.plot(mode, [0.8565, 0.8548])
# axis2.plot(mode, [0.8725, 0.8729])
# axis2.plot(mode, [0.7113, 0.7010])
# axis2.plot(mode, [0.6280, 0.6074])
# axis2.plot(mode, [0.8257, 0.8151])
# axis2.set_title("Support Vector Machine - f1 macro")
# figure.legend(facecolor="white", framealpha=0, loc="center right")

# # lower left
# axis3.plot(mode, [0.7646, 0.7710])
# axis3.plot(mode, [0.7687, 0.7631])
# axis3.plot(mode, [0.5669, 0.5925])
# axis3.plot(mode, [0.4450, 0.4637])
# axis3.plot(mode, [0.5600, 0.5923])
# axis3.set_title("Random Forest - f1 macro")

# figure.legend(facecolor="white", framealpha=0, loc="center right")

# # Setting the values for all axes.
# plt.setp((axis1, axis2, axis3), ylim=(0, 1))
# plt.show()
