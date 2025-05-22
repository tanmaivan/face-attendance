# Face Attendance System

Hệ thống điểm danh tự động sử dụng công nghệ nhận diện khuôn mặt, cho phép:

-   Đăng ký người dùng

-   Nhận diện khuôn mặt thời gian thực

-   Ghi nhận điểm danh

-   Quản lý qua bảng điều khiển thống kê chi tiết

---

## Tính năng chính

-   Đăng ký khuôn mặt: Thêm người dùng mới với ID và tên

-   Nhận diện khuôn mặt: Thời gian thực bằng webcam

-   Điểm danh tự động: Ghi nhận thời gian vào cơ sở dữ liệu

-   Bảng điều khiển: Thống kê chi tiết dữ liệu điểm danh

-   Quản lý người dùng: Xem và xóa người dùng đã đăng ký

-   Bảo mật: Hệ thống đăng nhập bảo vệ thông tin quản trị

---

## Yêu cầu hệ thống

-   Python ≥ 3.7

-   Camera (webcam) hoạt động tốt

-   CPU/GPU đủ mạnh để xử lý nhận diện khuôn mặt

---

## Cài đặt

### 1\. Clone repository

```
git clone https://github.com/tanmaivan/face-attendance.git
cd face-attendance
```

### 2\. Tạo môi trường ảo (khuyến nghị)

```

# Tạo môi trường
python -m venv venv

# Kích hoạt môi trường
# Windows
venv\Scripts\activate
# Linux/Mac
source venv/bin/activate
```

### 3\. Cài đặt dependencies

```
pip install -r requirements.txt
```

### 4\. Cấu hình tài khoản admin

Tạo file `cred.csv` với nội dung sau:

```
username,password
admin,admin123
```

Bạn có thể thay đổi username và password theo ý muốn.

### 5\. Khởi chạy ứng dụng

```
python app.py
```

Sau đó truy cập địa chỉ: `http://127.0.0.1:5000` trên trình duyệt.

---

## Cách sử dụng

### Đăng nhập quản trị

1.  Truy cập trang chủ

2.  Nhấn "Admin Login"

3.  Nhập thông tin đăng nhập từ file `cred.csv`

### Đăng ký khuôn mặt mới

1.  Đăng nhập với tài khoản admin

2.  Chọn "Đăng ký người dùng mới"

3.  Nhập ID và tên người dùng

4.  Thực hiện chụp ảnh theo hướng dẫn

### Nhận diện và điểm danh

1.  Chọn "Nhận diện khuôn mặt"

2.  Hệ thống sẽ quét khuôn mặt và ghi nhận điểm danh tự động

### Xem báo cáo điểm danh

1.  Đăng nhập với tài khoản admin

2.  Chọn "Xem điểm danh hôm nay" hoặc "Xem toàn bộ dữ liệu"

3.  Xem bảng thống kê chi tiết

---

## Cấu trúc mã nguồn

| File/Folder    | Mô tả                                                   |
| -------------- | ------------------------------------------------------- |
| `app.py`       | Ứng dụng Flask và các route chính                       |
| `face_func.py` | Các hàm xử lý khuôn mặt (thêm, xóa, nhận diện)          |
| `models.py`    | Mô hình Deep Learning để trích xuất embedding khuôn mặt |
| `settings.py`  | Các cài đặt và tham số hệ thống                         |
| `utils.py`     | Các hàm tiện ích cho việc quản lý cơ sở dữ liệu         |
| `templates/`   | Các file HTML cho giao diện người dùng                  |

---

## Tham số cấu hình (trong `settings.py`)

| Biến           | Mô tả                                        | Mặc định |
| -------------- | -------------------------------------------- | -------- |
| `THRESHOLD`    | Ngưỡng độ tin cậy cho nhận diện khuôn mặt    | 0.6      |
| `WIN_SIZE`     | Số khung hình liên tiếp để xác nhận          | 7        |
| `MIN_VOTES`    | Số phiếu tối thiểu để xác nhận một khuôn mặt | 5        |
| `IMG_PER_USER` | Số lượng ảnh cần lưu cho mỗi người dùng      | 50       |

---

## Bảo trì

### Xóa cơ sở dữ liệu

```
python face_func.py --mode remove_db
```

### Sao lưu dữ liệu

-   Dữ liệu điểm danh được lưu trong file `information.db`

-   Nên sao lưu file này thường xuyên để tránh mất mát

---

## Lưu ý

-   Hệ thống cần đủ ánh sáng để nhận diện khuôn mặt hiệu quả

-   Nên dùng ít nhất 30--50 ảnh cho mỗi người để tăng độ chính xác

-   Sau khi cài đặt dependencies, bạn có thể chạy ngay `python app.py` mà không cần cấu hình thêm

-   Tài khoản admin được cấu hình trong file `cred.csv`

---

## Công nghệ sử dụng

-   Flask -- Framework web

-   OpenCV -- Xử lý hình ảnh

-   MTCNN -- Phát hiện khuôn mặt

-   ResNet50d -- Trích xuất đặc trưng khuôn mặt

-   FAISS -- Lưu trữ và tìm kiếm vector embedding

-   SQLite -- Cơ sở dữ liệu lưu trữ điểm danh
