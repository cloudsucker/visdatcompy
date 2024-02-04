import pandas as pd
from PIL import Image
from PIL.ExifTags import TAGS
#from IPython.display import display
import os.path


#Получение exif из файла
def get_exif_data(image_path):
    #exif_list = []
    exif_list = {}
    try:
        with Image.open(image_path) as img:
            exif_data = img._getexif()
            if exif_data:
                for tag, value in exif_data.items():
                    tag_name = TAGS.get(tag, tag)
                    exif_list[tag_name] = value
            else:
                print("No EXIF data found.")
    except FileNotFoundError:
        print("File not found.")
        return
    
    return exif_list

#Получение списка файлов из директории
def get_files_from_directory(dir):
    filesList = []
    for address, dirs, files in os.walk(dir):
        for name in files:
            #проверка на формат файла
            if name.endswith('jpg'):
                filename, file_extension = os.path.splitext(name)
                filesList.append((filename, file_extension, os.path.join(address, name)))

    return filesList

#Сбор exif тегов со всех файлов
def get_exif_from_files(dir):
    filesExifList = []
    fileList = get_files_from_directory(dir)
    for file in fileList:
        #filesExifList.append(({'Filename': file[0]} | {'FileExtension': file[1]} | get_exif_data(file[2])))
        exif_data = get_exif_data(file[2])
        if exif_data is not None: 
            exif_data.update({'Filename': file[0], 'FileExtension': file[1]})
            filesExifList.append(exif_data)

    return filesExifList

#Создание датафрейма exif из директории
def create_exif_dataframe(dataPath):
    #dataPath = 'C:/Users/UserPC/archive'
    data = get_exif_from_files(dataPath)
    #exiftags = list(TAGS.values())
    df = pd.DataFrame(data = data,)
    df = df.drop(columns='MakerNote')
    df.to_csv('meta.csv', header=True, index=True)

    return df
    #display(df)