import os
import shutil


class ConvertTestFolder:
    def __init__(self):
        pass

    def format_file_name(self, category_id, index, extension):
        return f"{category_id}.{index}.{extension}"

    def change_name(self, folder_name, index, file_name):
        extension = file_name.split('.')[-1].lower()
        if extension not in ['jpg', 'png']:
            print(f"Invalid file extension: {file_name}")
            return None
        category_mapping = {
            'nhan': '11203',
            'vong_tay': '11202',
            'bong_tai': '11205',
            'day_chuyen': '11208',
            'charm': '11211',
            'kieng': '11212',
            'lac': '27601',
        }
        category_id = category_mapping.get(folder_name)
        if not category_id:
            print(f"Invalid folder name: {folder_name}")
            return None
        return self.format_file_name(category_id, index, extension)

    def convert(self, folder_path, folder_goal):
        if not os.path.exists(folder_path):
            raise FileNotFoundError(f"Source folder does not exist: {folder_path}")
        if not os.path.exists(folder_goal):
            os.makedirs(folder_goal)

        global_index = 1  # Dùng để đánh số thứ tự duy nhất cho tất cả các file
        for folder_item in os.listdir(folder_path):
            source_folder = os.path.join(folder_path, folder_item)
            if os.path.isdir(source_folder):
                for file_name in os.listdir(source_folder):
                    if file_name.lower().endswith((".jpg", ".png")):
                        new_name = self.change_name(folder_item, global_index, file_name)
                        if new_name:
                            source_file = os.path.join(source_folder, file_name)
                            target_file = os.path.join(folder_goal, new_name)
                            shutil.copy2(source_file, target_file)
                            print(f"Copied and renamed: {source_file} -> {target_file}")
                            global_index += 1


def main():
    convert_test_folder = ConvertTestFolder()
    folder = r'E:/Images/khoi'  # Thư mục gốc
    folder_goal = r'E:/Images/test'  # Thư mục đích
    convert_test_folder.convert(folder, folder_goal)


if __name__ == '__main__':
    main()
