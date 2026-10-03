import os
import numpy as np

# Thiết lập đường dẫn gốc dự án và các thư mục lưu dữ liệu
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TEST_DATA_DIR = os.path.join(BASE_DIR, "test_data")
RESULTS_DIR = os.path.join(BASE_DIR, "results")

# Tự động tạo thư mục lưu trữ nếu chưa có
os.makedirs(TEST_DATA_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# Tham số hệ thống: tần số lấy mẫu fs = 8000 Hz, độ dài xung, độ trễ và chiều dài tín hiệu
FS = 8000                  # Tần số lấy mẫu (Hz): 1 mẫu = 1/8000 s = 0.125 ms
PULSE_LEN = 50             # Chiều dài xung mặc định (50 mẫu = 6.25 ms)
TRUE_DELAY = 200           # Độ trễ thực tế mặc định (200 mẫu = 25.0 ms)
SIG_LEN = 500              # Chiều dài tín hiệu thu quan sát (500 mẫu = 62.5 ms)
PEAK_AMPLITUDE = 1.0       # Biên độ đỉnh xung phát chuẩn

# Định nghĩa dung sai (Tolerance) theo đề bài (≤ 2 ms) và quy đổi sang mẫu
TOLERANCE_MS = 2.0         # Ngưỡng dung sai thời gian theo đề bài (2.0 ms)
TOLERANCE = int(round(TOLERANCE_MS * FS / 1000.0))  # 2.0 ms * 8000 Hz / 1000 = 16 mẫu
TOLERANCE_STRICT = 2       # Ngưỡng dung sai chặt (2 mẫu = 0.25 ms) dùng để so sánh nâng cao

# Dải SNR từ -15 dB đến 15 dB và cấu hình số lần thử Monte Carlo
SNR_MIN_DB = -15
SNR_MAX_DB = 15
SNR_STEP_DB = 1
SNR_RANGE_DB = np.arange(SNR_MIN_DB, SNR_MAX_DB + 1, SNR_STEP_DB)
N_TRIALS = 100
MC_SEED = 2024             # Seed cố định để mô phỏng Monte Carlo 100% tái lập được

# Cấu hình 4 kịch bản kiểm thử đại diện cho 4 tình huống vật lý (Ví dụ minh họa có kiểm soát seed)
# Ghi chú: Các seed được cố định nhằm minh họa trực quan 4 trường hợp điển hình:
# - Test 1: Bắt trúng tuyệt đối ở SNR cao
# - Test 2: Minh họa xê dịch nhẹ do đỉnh tù ở vùng chuyển tiếp (-5 dB)
# - Test 3: Minh họa hiệu quả tăng năng lượng xung Es khi xung dài hơn
# - Test 4: Minh họa trường hợp đỉnh giả ở xa tại SNR cực thấp (-10 dB)
# (Đánh giá thống kê tổng thể khách quan toàn diện được thực hiện ở Bước 3 qua mô phỏng Monte Carlo)
TEST_CASES = {
    # Kịch bản 1: Mức SNR cao (+10 dB), lý tưởng, bắt trúng tuyệt đối D = 200
    "test1": {
        "name": "Kịch bản 1: SNR cao (Lý tưởng, bắt trúng)",
        "pulse_len": 50,
        "true_delay": 200,
        "sig_len": 500,
        "snr_db": 10.0,
        "seed": 2024,
        "file_name": "test1.npz",
        "fig_name": "fig1_test1.png",
        "corner": "top-left"
    },
    # Kịch bản 2: Mức SNR thấp (-5 dB), vùng chuyển tiếp, đỉnh thật bị xê dịch nhẹ do đỉnh tù
    "test2": {
        "name": "Kịch bản 2: SNR thấp (Minh họa xê dịch nhẹ do đỉnh tù)",
        "pulse_len": 50,
        "true_delay": 200,
        "sig_len": 500,
        "snr_db": -5.0,
        "seed": 99,
        "file_name": "test2.npz",
        "fig_name": "fig2_test2.png",
        "corner": "top-right"
    },
    # Kịch bản 3: Xung dài hơn (80 mẫu) giúp tăng năng lượng Es, trễ thay đổi thành 150 mẫu
    "test3": {
        "name": "Kịch bản 3: Độ trễ khác & Xung dài hơn (Tăng năng lượng Es, đỉnh tù hơn)",
        "pulse_len": 80,
        "true_delay": 150,
        "sig_len": 600,
        "snr_db": 5.0,
        "seed": 101,
        "file_name": "test3.npz",
        "fig_name": "fig3_test3.png",
        "corner": "bottom-left"
    },
    # Kịch bản 4: Xung ngắn (35 mẫu) ở SNR cực thấp (-10 dB), nhiễu tạo đỉnh giả ở xa
    "test4": {
        "name": "Kịch bản 4: Xung ngắn, SNR cực thấp (Minh họa đỉnh giả ở xa)",
        "pulse_len": 35,
        "true_delay": 280,
        "sig_len": 550,
        "snr_db": -10.0,
        "seed": 999,
        "file_name": "test4.npz",
        "fig_name": "fig4_test4.png",
        "corner": "bottom-right"
    }
}
