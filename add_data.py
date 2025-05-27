import sqlite3
from datetime import datetime, timedelta
import random

# Danh sách tên học sinh
names = [
    "Nguyen Van A", "Tran Thi B", "Le Van C", "Pham Thi D", "Hoang Van E",
    "Vu Thi F", "Dang Van G", "Bui Thi H", "Do Van I", "Ngo Thi K",
    "Ly Van L", "Truong Thi M", "Nguyen Van N", "Tran Thi O", "Le Van P"
]

# Kết nối đến database
conn = sqlite3.connect('information.db')
cursor = conn.cursor()

# Lấy thời gian hiện tại
now = datetime.now()

# Thời gian điểm danh buổi sáng và chiều
morning_times = ["07:30", "07:45", "08:00", "08:15"]
afternoon_times = ["13:30", "13:45", "14:00", "14:15"]

# Thêm dữ liệu cho mỗi học sinh
for name in names:
    # Mỗi học sinh sẽ có điểm danh trong 15 ngày gần đây
    for day in range(15):
        # 80% khả năng học sinh đi học
        if random.random() < 0.8:
            # Chọn ngẫu nhiên 1-2 lần điểm danh mỗi ngày
            num_attendance = random.randint(1, 2)
            
            for _ in range(num_attendance):
                # Chọn ngẫu nhiên buổi sáng hoặc chiều
                if random.random() < 0.5:
                    time_str = random.choice(morning_times)
                else:
                    time_str = random.choice(afternoon_times)
                
                # Tính ngày điểm danh
                date = now - timedelta(days=day)
                date_str = date.strftime("%Y-%m-%d")
                
                # Thêm dữ liệu vào database
                cursor.execute(
                    "INSERT INTO Attendance (Name, Time, Date) VALUES (?, ?, ?)",
                    (name, time_str, date_str)
                )

# Lưu thay đổi và đóng kết nối
conn.commit()
conn.close()

print("Đã thêm dữ liệu điểm danh mới vào database.")