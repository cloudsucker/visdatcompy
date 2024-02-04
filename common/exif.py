import pandas as pd
from PIL import Image
from PIL.ExifTags import TAGS
#from IPython.display import display
import os.path
from utils import scan_directory

#Получение exif из файла
def get_exif_data(image_path) -> list:
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

#Сбор exif тегов с файлов из директории
def get_exif_from_files(dir) -> list:
    filesExifList = []
    fileList = scan_directory(dir)
    for file in fileList:
        exif_data = get_exif_data(os.path.join(file[0],file[1]))
        if exif_data is not None: 
            filename, file_extension = os.path.splitext(file[1])
            exif_data.update({'Filename': filename, 'FileExtension': file_extension})
            filesExifList.append(exif_data)
            print(exif_data)
    return filesExifList

#Создание датафрейма exif из директории
def create_exif_dataframe(dataPath) -> pd.DataFrame:
    data = get_exif_from_files(dataPath)
    df = pd.DataFrame(data = data,)
    df = df.drop(columns='MakerNote')
    df.to_csv('meta.csv', header=True, index=True)

    return df
    #display(df)