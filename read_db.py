import sqlite3
import pandas as pd

# Kết nối đến database
conn = sqlite3.connect('information.db')

# Đọc tất cả các bảng trong database
tables = pd.read_sql("SELECT name FROM sqlite_master WHERE type='table';", conn)
print("Các bảng trong database:")
print(tables)

# Đọc dữ liệu từ mỗi bảng
for table in tables['name']:
    print(f"\nDữ liệu trong bảng {table}:")
    df = pd.read_sql(f"SELECT * FROM {table}", conn)
    print(df)

conn.close() 