from zipfile import ZipFile
from datetime import date
import os

def get_all_file_paths(dir):
    file_paths = []

    for root, dirs, files, in os.walk(dir):
        for filename in files:
            filepath = os.path.join(root, filename)
            file_paths.append(filepath)

    return file_paths

def main():
    directory = './RESULT DATA'
    file_paths = get_all_file_paths(directory)

    print("Following files will be zipped: ")
    for file_name in file_paths:
        print(file_name)
    
    today = str(date.today())
    zip_name = 'SUBMISSIONS/lm2491_D2-D6_' + today
    with ZipFile(zip_name, 'w') as zip:
        for file in file_paths:
            zip.write(file)

    print("All files zipped.")

if __name__ == "__main__":
    main()