import sqlite3

# Kết nối đến database
conn = sqlite3.connect('information.db')
cursor = conn.cursor()

# Xóa tất cả dữ liệu trong bảng Attendance
cursor.execute("DELETE FROM Attendance")

# Lưu thay đổi và đóng kết nối
conn.commit()
conn.close()

print("Đã xóa dữ liệu cũ trong database.")