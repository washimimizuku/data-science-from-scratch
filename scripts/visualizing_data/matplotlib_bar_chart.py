from matplotlib import pyplot as plt

movies = ["Annie Hall", "Ben-Hur", "Casablanca", "Gandhi", "West Side Story"]
num_oscars = [5, 11, 3, 8, 10]

# Plot bars with left x-coordinates [0, 1, 2, 3, 4], heights [num_oscars]
plt.bar(range(len(movies)), num_oscars)

plt.title("My Favorite Movies") # Add a title
plt.ylabel("# of Academy Awards") # Label the y-axis
plt.xticks(range(len(movies)), movies) # Label x-axis with movie names at bar centers

plt.show()