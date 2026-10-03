import numpy as np

def compute_cross_correlation(reference: np.ndarray, received: np.ndarray):
    """
    Chức năng: Tính hàm tương quan chéo r_sr[ℓ] = sum(s[n] * r[n + ℓ]) giữa xung s[n] và tín hiệu thu r[n].
    Đầu vào:
      - reference (np.ndarray): Mẫu xung phát chuẩn s[n], chiều dài M mẫu.
      - received (np.ndarray): Tín hiệu thu r[n], chiều dài N mẫu.
    Đầu ra:
      - corr (np.ndarray): Mảng chứa các giá trị tương quan chéo tại từng độ lệch lag ℓ.
      - lags (np.ndarray): Mảng chỉ số lag ℓ tương ứng (từ -(M-1) đến N-1).
    """
    M = len(reference)
    N = len(received)
    
    # Xác định dải dịch lag từ -(M - 1) đến +(N - 1) với tổng cộng M + N - 1 điểm lag
    min_lag = -(M - 1)
    max_lag = N - 1
    num_lags = max_lag - min_lag + 1
    lags = np.zeros(num_lags, dtype=int)
    corr = np.zeros(num_lags, dtype=float)
    
    # Trượt xung tham chiếu s[n] dọc theo r[n] và tính tích vô hướng tại từng bước dịch lag ℓ
    for idx, lag in enumerate(range(min_lag, max_lag + 1)):
        lags[idx] = lag
        n_min = max(0, -lag)
        n_max = min(M - 1, N - 1 - lag)
        
        # Tích lũy tích vô hướng khi hai tín hiệu có vùng gối lên nhau hợp lệ: r_sr[ℓ] = sum(s[n] * r[n + ℓ])
        if n_min <= n_max:
            s_slice = reference[n_min : n_max + 1]
            r_slice = received[n_min + lag : n_max + 1 + lag]
            corr[idx] = np.sum(s_slice * r_slice)
            
    return corr, lags

def find_peak_delay(corr: np.ndarray, lags: np.ndarray) -> int:
    """
    Chức năng: Dò tìm vị trí đỉnh cực đại lớn nhất của hàm tương quan chéo r_sr[ℓ] với trễ vật lý ℓ >= 0.
    Đầu vào:
      - corr (np.ndarray): Mảng giá trị tương quan chéo.
      - lags (np.ndarray): Mảng chỉ số độ lệch lag tương ứng.
    Đầu ra:
      - best_lag (int): Độ trễ ước lượng D̂ (đơn vị: mẫu).
    """
    max_val = -1e30
    best_lag = 0
    
    # Duyệt qua các độ trễ không âm để tìm vị trí đỉnh dương lớn nhất của hàm tương quan chéo
    for i in range(len(lags)):
        lag = lags[i]
        if lag >= 0:
            val = corr[i]
            if val > max_val:
                max_val = val
                best_lag = lag
                
    return int(best_lag)

def estimate_delay(reference: np.ndarray, received: np.ndarray):
    """
    Chức năng: Hàm tích hợp thực hiện tính tương quan chéo và dò tìm độ trễ ước lượng.
    Đầu vào:
      - reference (np.ndarray): Xung phát tham chiếu s[n].
      - received (np.ndarray): Tín hiệu thu r[n].
    Đầu ra:
      - estimated_delay (int): Độ trễ ước lượng D̂.
      - corr (np.ndarray): Mảng hàm tương quan chéo r_sr[ℓ].
      - lags (np.ndarray): Mảng chỉ số độ lệch lag.
    """
    corr, lags = compute_cross_correlation(reference, received)
    estimated_delay = find_peak_delay(corr, lags)
    return estimated_delay, corr, lags
