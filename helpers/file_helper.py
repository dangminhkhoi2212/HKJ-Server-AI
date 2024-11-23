import os
import pickle


def save_file(file, path):
    try:
        # Kiểm tra nếu file đã tồn tại
        if os.path.exists(path):
            os.remove(path)  # Xóa file cũ
            print(f"Deleted existing file: {path}")

        # Lưu file mới
        with open(path, "wb") as f:
            pickle.dump(file, f)
        print(f"Successfully saved {path} file")

    except Exception as e:
        print(f"Error saving file to {path}: {str(e)}")


def load_file(path):
    try:
        with open(path, "rb") as f:
            data = pickle.load(f)
        print(f"Successfully loaded data from {path}")
        return data
    except Exception as e:
        print(f"Error loading pickle file: {str(e)}")
        return None
