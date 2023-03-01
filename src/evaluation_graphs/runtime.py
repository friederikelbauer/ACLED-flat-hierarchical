import matplotlib.pyplot as plt
import scienceplots

plt.style.use("science")

# data
mode = ["flat", "hierarchical"]

# Initialise the subplot function using number of rows and columns
figure, (axis1, axis2, axis3) = plt.subplots(nrows=1, ncols=3)

# upper left
axis1.plot(mode, [1379, 746], label="Count Vectorizer")
axis1.plot(mode, [511, 287], label="TF-IDF")
axis1.plot(mode, [415, 319], label="Word2Vec")
axis1.plot(mode, [861, 851], label="Fast Text")
axis1.set_title("Logistic Regression")
axis1.set(ylabel="seconds")

# upper right
axis2.plot(mode, [269, 236])
axis2.plot(mode, [77, 78])
axis2.plot(mode, [398, 283])
axis2.plot(mode, [992, 944])
axis2.set_title("Support Vector Machine")

# lower left
axis3.plot(mode, [7414, 8308])
axis3.plot(mode, [5822, 6993])
axis3.plot(mode, [1265, 1954])
axis3.plot(mode, [1680, 2154])
axis3.set_title("Random Forest")
figure.legend(facecolor="white", framealpha=0, loc="center right")
plt.show()


# BERT visualization
fig, ax = plt.subplots()
ax.plot(mode, [5696, 2582], label="Logistic Regression")
ax.plot(mode, [1785, 1239], label="Support Vector Machine")
ax.plot(mode, [2796, 4255], label="Random Forest")
ax.set_title("BERT Vectorizer comparison")
ax.legend()
plt.show()
