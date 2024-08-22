import pickle


def save_file(file, path):
    try:
        pickle.dump(file, open(path, "wb"))
        print(f"Successfully saved {path} file")
    except Exception as e:
        print(f"Error saving file: {str(e)}")


def load_file(path):
    try:
        file = pickle.load(open(path, "rb"))
        print(f"Successfully loaded {path} file")
        return file
    except Exception as e:
        print(f"Error loading file: {str(e)}")
        return None
