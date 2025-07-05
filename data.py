import pickle

# Load the pickle file
with open("data.pickle", "rb") as f:
    data = pickle.load(f)

# Print to understand the structure
print(type(data))  # Check if it's a list, dict, or something else
print(data)        # Print the content (if it's not too large)
