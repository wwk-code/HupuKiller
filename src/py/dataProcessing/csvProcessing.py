import pandas as pd



# 指定 CSV 文件的路径
csv_file_path = 'path/to/your/file.csv'

# 读取 CSV 文件
try:
    data = pd.read_csv(csv_file_path)

    # 打印读取的数据
    print(data)
except FileNotFoundError:
    print(f"文件未找到: {csv_file_path}")
except pd.errors.EmptyDataError:
    print("文件是空的")
except pd.errors.ParserError:
    print("文件解析错误")