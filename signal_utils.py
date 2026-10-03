import numpy as np

def generate_triangle_pulse(length: int, amplitude: float = 1.0) -> np.ndarray:
    """
    Chức năng: Tạo xung tam giác đối xứng s[n] có đỉnh cực đại amplitude tại tâm xung.
    Đầu vào:
      - length (int): Chiều dài xung (số mẫu M).
      - amplitude (float): Biên độ cực đại lý thuyết tại đỉnh xung (mặc định: 1.0).
    Đầu ra:
      - pulse (np.ndarray): Mảng 1 chiều chứa các mẫu của xung tam giác s[n].
    Ghi chú kỹ thuật:
      - Với M chẵn (ví dụ M=50), tâm xung nằm tại vị trí phân số center = (M - 1)/2 = 24.5.
        Do đó đỉnh thực tế đạt tại n=24 và n=25 là 1.0 * (1 - 0.5/24.5) ≈ 0.9796.
      - Hai mút s[0] = s[M-1] = 0, nên xung hiệu dụng chứa năng lượng có chiều dài M - 2 = 48 mẫu.
    """
    pulse = np.zeros(length, dtype=float)
    center = (length - 1) / 2.0
    
    # Tính giá trị từng mẫu theo hàm tam giác đối xứng chuẩn: s[n] = A * (1 - |n - center| / center)
    for n in range(length):
        if center > 0:
            val = amplitude * (1.0 - abs(n - center) / center)
            pulse[n] = max(0.0, val)
        else:
            pulse[n] = amplitude
            
    return pulse

def create_received_signal(pulse: np.ndarray, delay: int, total_len: int) -> np.ndarray:
    """
    Chức năng: Tạo tín hiệu thu lý tưởng r_clean[n] = s[n - D] không nhiễu.
    Đầu vào:
      - pulse (np.ndarray): Mẫu xung phát s[n].
      - delay (int): Độ trễ thực tế D (số mẫu).
      - total_len (int): Tổng số mẫu quan sát của tín hiệu thu.
    Đầu ra:
      - r_clean (np.ndarray): Tín hiệu thu chứa xung phát dịch trễ D mẫu.
    """
    r_clean = np.zeros(total_len, dtype=float)
    pulse_len = len(pulse)
    
    # Đặt các mẫu của xung s[n] vào vị trí tương ứng từ index delay đến delay + pulse_len
    for i in range(pulse_len):
        idx = delay + i
        if 0 <= idx < total_len:
            r_clean[idx] = pulse[i]
            
    return r_clean

def compute_energy(signal: np.ndarray) -> float:
    """
    Chức năng: Tính năng lượng toàn phần của tín hiệu rời rạc Es = sum(|x[n]|^2).
    Đầu vào:
      - signal (np.ndarray): Tín hiệu x[n] cần đo năng lượng.
    Đầu ra:
      - energy (float): Tổng năng lượng tích lũy của tín hiệu.
    """
    energy = 0.0
    # Tích lũy tổng bình phương của từng mẫu tín hiệu
    for val in signal:
        energy += float(val * val)
    return energy

def add_awgn(signal: np.ndarray, snr_db: float, peak_amplitude: float = 1.0, seed: int = None, rng: np.random.RandomState = None) -> np.ndarray:
    """
    Chức năng: Cộng nhiễu trắng Gauss (AWGN) vào tín hiệu theo mức Peak SNR mong muốn.
    Định nghĩa SNR:
      - Hệ thống sử dụng Peak SNR: SNR_peak_dB = 10 * log10(A_peak^2 / sigma^2).
      - Quy đổi tương đương đối với xung tam giác (M=50, Es ≈ 16.33):
        + Công suất trung bình xung / sigma^2 = SNR_peak_dB + 10*log10(Es / (M * A^2)) ≈ SNR_peak_dB - 4.86 dB.
        + Matched Filter Output SNR = Es / sigma^2 = SNR_peak_dB + 10*log10(Es / A^2) ≈ SNR_peak_dB + 12.13 dB.
    Đầu vào:
      - signal (np.ndarray): Tín hiệu sạch đầu vào.
      - snr_db (float): Tỷ số Peak SNR mong muốn tính theo dB.
      - peak_amplitude (float): Biên độ đỉnh của xung dùng làm tham chiếu tính SNR.
      - seed (int, tùy chọn): Hạt giống số ngẫu nhiên để tái lập kết quả thử nghiệm.
      - rng (np.random.RandomState, tùy chọn): Đối tượng sinh số ngẫu nhiên dùng chung.
    Đầu ra:
      - noisy_signal (np.ndarray): Tín hiệu thu sau khi bị cộng nhiễu.
    """
    if rng is None:
        if seed is not None:
            rng = np.random.RandomState(seed)
        else:
            rng = np.random.RandomState()
        
    # Tính phương sai và độ lệch chuẩn của nhiễu theo công thức SNR = 10*log10(A_peak^2 / sigma^2)
    sigma2 = (peak_amplitude ** 2) / (10.0 ** (snr_db / 10.0))
    sigma = np.sqrt(sigma2)
    noise = rng.randn(len(signal)) * sigma
    
    # Cộng từng mẫu tín hiệu sạch với mẫu nhiễu tương ứng
    noisy_signal = np.zeros(len(signal), dtype=float)
    for i in range(len(signal)):
        noisy_signal[i] = signal[i] + noise[i]
        
    return noisy_signal

def save_test_data(filepath: str, data_dict: dict) -> None:
    """
    Chức năng: Lưu gói dữ liệu kiểm thử ra tệp định dạng chuẩn .npz (hoặc .npy).
    Đầu vào:
      - filepath (str): Đường dẫn lưu tệp (.npz).
      - data_dict (dict): Dictionary chứa các mảng tín hiệu và tham số kịch bản.
    Đầu ra:
      - None.
    """
    if filepath.endswith('.npz'):
        np.savez_compressed(filepath, **data_dict)
    else:
        np.save(filepath, data_dict, allow_pickle=True)

def load_test_data(filepath: str) -> dict:
    """
    Chức năng: Đọc gói dữ liệu kiểm thử từ tệp định dạng .npz (hoặc .npy).
    Đầu vào:
      - filepath (str): Đường dẫn tới tệp cần đọc.
    Đầu ra:
      - data (dict): Dictionary chứa tín hiệu phát, tín hiệu thu và thông số kịch bản.
    """
    if filepath.endswith('.npz'):
        with np.load(filepath, allow_pickle=False) as loaded:
            res = {}
            for k in loaded.files:
                val = loaded[k]
                if val.ndim == 0:
                    res[k] = val.item()
                else:
                    res[k] = val
            return res
    else:
        return np.load(filepath, allow_pickle=True).item()
