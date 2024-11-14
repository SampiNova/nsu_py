from PIL import Image

def get_binary_string_from_bmp(bmp_file_path):
 """
 Получает двоичную строку из BMP-файла.

 Args:
  bmp_file_path: Путь к BMP-файлу.

 Returns:
  Двоичная строка, представляющая содержимое BMP-файла.
 """

 with open(bmp_file_path, "rb") as f:
  # Читаем весь файл в двоичном виде
  binary_data = f.read()

 # Преобразуем двоичные данные в строку
 binary_string = ''.join(f"{byte:08b}" for byte in binary_data)

 return binary_string

# Пример использования
bmp_file_path = "Nothing.bmp" # Замените на путь к вашему файлу
binary_string = get_binary_string_from_bmp(bmp_file_path)

print(binary_string)

res = []
for elem in binary_string.split("0"):
    if elem != "":
        res.append(elem)

print(res)
