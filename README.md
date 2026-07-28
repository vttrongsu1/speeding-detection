# 🚗 Hệ thống Đo Tốc độ Xe chạy Thời gian Thực (Real-time Vehicle Speed Detection)

Ứng dụng web viết bằng Python (Flask) tích hợp mô hình YOLOv8 và thư viện Supervision để phát hiện phương tiện, bám vết (tracking) và đo tốc độ xe chạy thực tế trên từng làn đường bằng hiệu chuẩn phối cảnh (Perspective Transformation).

---

## 🛠️ Hướng dẫn cài đặt các thư viện cần thiết

Để chạy dự án này, máy tính của bạn cần cài đặt sẵn **Python** (Khuyên dùng Python từ phiên bản **3.8 đến 3.10**). 

Mở Terminal (Command Prompt hoặc PowerShell) và chạy lệnh sau để cài đặt toàn bộ các thư viện cần thiết:

```bash
# Cài đặt Flask, mô hình AI YOLOv8, thư viện Supervision và thư viện xử lý ảnh OpenCV, NumPy
pip install flask ultralytics supervision opencv-python numpy
```

---

## 🚀 Hướng dẫn cách khởi chạy ứng dụng

1. Mở Terminal (Command Prompt hoặc PowerShell).
2. Di chuyển đường dẫn terminal vào thư mục chứa dự án:
   ```bash
   cd đường_dẫn_đến_thư_mục_chứa_code
   ```
3. Khởi chạy máy chủ Web Backend:
   ```bash
   python app.py
   ```
4. Mở trình duyệt web bất kỳ và truy cập địa chỉ:
   ```
   http://localhost:5000
   ```

---

## 📖 Hướng dẫn sử dụng giao diện Web chi tiết

Khi giao diện bảng điều khiển hiện ra trên trình duyệt, bạn thực hiện theo các bước sau để đo tốc độ:

### Bước 1: Tải video lên hệ thống
* Nhấp vào nút **`CHỌN VÀ TẢI LÊN VIDEO`** lớn ở phía trên để mở trình quản lý tệp tin.
* Chọn bất kỳ tệp video nào từ máy tính của bạn (hỗ trợ tệp lên đến 500MB).
* Sau khi tải lên thành công, hệ thống sẽ tự động hiển thị khung hình đầu tiên của video để bạn thực hiện căn chỉnh.

### Bước 2: Thiết lập và hiệu chuẩn các vùng giám sát (Vùng đo tốc độ)
* **Khởi tạo**: Ban đầu hệ thống sẽ ở trạng thái trống hoàn toàn không có tọa độ vẽ sẵn.
* **Tạo vùng đo**: Nhấp vào nút **`➕ THÊM VÙNG MỚI (TẠO TAG MỚI)`** ở bên dưới. Vùng mới sẽ được tạo (Ví dụ: Vùng 1).
* **Vẽ tọa độ**: Click lần lượt **4 điểm** trên khung ảnh video theo thứ tự hình thang:
  $$\text{Top-Left (Trên-Trái)} \rightarrow \text{Top-Right (Trên-Phải)} \rightarrow \text{Bottom-Right (Dưới-Phải)} \rightarrow \text{Bottom-Left (Dưới-Trái)}$$
  *(Bạn có thể nhấn nút "Hoàn tác điểm" nếu nhấp nhầm hoặc nhấp "Xóa vùng hiện tại" để vẽ lại).*
* **Điền thông số của vùng**: Nhập đúng các thông số thực tế của đoạn đường bạn vừa khoanh:
  * **Rộng làn đường (mét)**: Chiều rộng thực của các làn đường nằm trong hình thang (Ví dụ: 11.0m cho 3 làn đường).
  * **Dài đoạn đường (mét)**: Khoảng cách thực tế từ biên trên tới biên dưới của hình thang (Ví dụ: 50m hoặc 100m).
  * **Tốc độ giới hạn (km/h)**: Giới hạn tốc độ cho phép của tuyến đường để phát hiện lỗi vượt tốc độ (Ví dụ: 60km/h).
* **Đo nhiều chiều đường (Đa Vùng - Multi-zone)**: Nếu muốn đo song song chiều đường ngược lại hoặc vị trí khác, tiếp tục nhấp **`➕ THÊM VÙNG MỚI`** để tạo thêm các tab riêng biệt (Vùng 2, Vùng 3...) và làm tương tự. Mỗi vùng sẽ được vẽ bằng một màu sắc viền độc lập.

### Bước 3: Khởi chạy và giám sát
* Nhấp nút **`▶ BẮT ĐẦU CHẠY (ĐO TỐC ĐỘ)`**.
* Luồng livestream xử lý AI trực tiếp sẽ chạy. Xe chạy qua từng vùng sẽ hiển thị khung bao nhận diện kèm mã số ID và tốc độ nhảy số liên tục theo đúng tỉ lệ phối cảnh của vùng đó.
* Các chỉ số (Số xe trong vùng, Tốc độ trung bình, Số xe vi phạm vượt tốc độ) và Nhật ký xe vi phạm sẽ cập nhật tức thì.
* **Dừng để chỉnh sửa**: Nếu muốn thay đổi tọa độ hay thông số khi đang chạy, bạn chỉ cần bấm **`📐 Dừng & Chỉnh Lại Tọa Độ`** để quay lại chế độ căn chỉnh, thay đổi thông số và nhấn Bắt đầu chạy lại mà không cần tải lại trang.
