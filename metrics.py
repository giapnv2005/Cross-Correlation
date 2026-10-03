import numpy as np
import csv
from signal_utils import create_received_signal, add_awgn
from correlation import estimate_delay

def compute_error(estimated_delay: int, true_delay: int) -> int:
    """
    Chức năng: Tính sai số tuyệt đối giữa độ trễ ước lượng và độ trễ thực tế.
    Đầu vào:
      - estimated_delay (int): Độ trễ ước lượng D̂ (mẫu).
      - true_delay (int): Độ trễ thực tế D (mẫu).
    Đầu ra:
      - (int): Sai số tuyệt đối |D̂ - D| tính bằng số mẫu.
    """
    return abs(estimated_delay - true_delay)

def calculate_rmse(errors: list) -> float:
    """
    Chức năng: Tính căn bậc hai sai số toàn phương trung bình (RMSE).
    Đầu vào:
      - errors (list hoặc np.ndarray): Danh sách các sai số ước lượng qua các lần thử.
    Đầu ra:
      - (float): Giá trị sai số RMSE.
    """
    if len(errors) == 0:
        return 0.0
    sum_sq = 0.0
    for e in errors:
        sum_sq += float(e * e)
    return float(np.sqrt(sum_sq / len(errors)))

def calculate_detection_rate(errors: list, tolerance: int = 2) -> float:
    """
    Chức năng: Tính tỷ lệ phát hiện chính xác (%) dựa trên ngưỡng dung sai cho phép.
    Đầu vào:
      - errors (list hoặc np.ndarray): Danh sách các sai số ước lượng.
      - tolerance (int): Dung sai sai số tối đa để coi là phát hiện đúng (mặc định: 2 mẫu).
    Đầu ra:
      - (float): Tỷ lệ phát hiện đúng (tính theo phần trăm 0% - 100%).
    """
    if len(errors) == 0:
        return 0.0
    correct_count = 0
    for e in errors:
        if e <= tolerance:
            correct_count += 1
    return float((correct_count / len(errors)) * 100.0)

def calculate_mean_error(errors: list) -> float:
    """
    Chức năng: Tính sai số trung bình (Mean Absolute Error - MAE).
    Đầu vào:
      - errors (list hoặc np.ndarray): Danh sách sai số tuyệt đối của các lần thử.
    Đầu ra:
      - (float): Giá trị sai số trung bình.
    """
    if len(errors) == 0:
        return 0.0
    total = 0.0
    for e in errors:
        total += float(e)
    return float(total / len(errors))

def compute_cramer_rao_bound(pulse: np.ndarray, snr_range_db: np.ndarray, peak_amp: float = 1.0) -> np.ndarray:
    """
    Chức năng: Tính giới hạn lý thuyết Cramér–Rao Bound (CRB) cho độ trễ theo từng mức SNR:
               CRB(D̂) = sqrt( sigma_w^2 / sum_{n}( (s[n+1] - s[n])^2 ) ) (đơn vị: mẫu).
    Đầu vào:
      - pulse (np.ndarray): Xung phát chuẩn s[n].
      - snr_range_db (np.ndarray): Mảng mức Peak SNR (dB).
      - peak_amp (float): Biên độ đỉnh xung phát.
    Đầu ra:
      - (np.ndarray): Mảng giá trị CRB (độ lệch chuẩn tối thiểu lý thuyết) theo từng mức SNR.
    """
    diff_s = np.diff(pulse)
    sum_diff_sq = np.sum(diff_s ** 2)
    if sum_diff_sq <= 0:
        return np.zeros_like(snr_range_db, dtype=float)
    
    crb_samples = np.zeros(len(snr_range_db), dtype=float)
    for i, snr in enumerate(snr_range_db):
        sigma2 = (peak_amp ** 2) / (10.0 ** (snr / 10.0))
        crb_samples[i] = np.sqrt(sigma2 / sum_diff_sq)
    return crb_samples

def run_monte_carlo(pulse: np.ndarray,
                     true_delay: int,
                     sig_len: int,
                     snr_range_db: np.ndarray,
                     n_trials: int = 100,
                     tolerance: int = 16,
                     tolerance_strict: int = 2,
                     peak_amp: float = 1.0,
                     fs: float = 8000.0,
                     seed: int = 2024,
                     verbose: bool = True) -> dict:
    """
    Chức năng: Thực hiện mô phỏng Monte Carlo qua toàn bộ dải SNR để thu thập thống kê hiệu năng.
    Đầu vào:
      - pulse (np.ndarray): Xung phát chuẩn s[n].
      - true_delay (int): Độ trễ thực tế D (số mẫu).
      - sig_len (int): Chiều dài tín hiệu thu.
      - snr_range_db (np.ndarray): Mảng các mức SNR kiểm thử (dB).
      - n_trials (int): Số lần lặp thử nghiệm ngẫu nhiên cho mỗi mức SNR (mặc định: 100).
      - tolerance (int): Dung sai thời gian quy đổi sang mẫu (mặc định: 16 mẫu = 2.0 ms tại fs=8000Hz).
      - tolerance_strict (int): Dung sai chặt để đối chiếu (mặc định: 2 mẫu = 0.25 ms).
      - peak_amp (float): Biên độ đỉnh xung phát.
      - fs (float): Tần số lấy mẫu hệ thống (Hz).
      - seed (int): Hạt giống ngẫu nhiên để tái lập kết quả 100%.
      - verbose (bool): Cờ bật/tắt in tiến trình mô phỏng ra console.
    Đầu ra:
      - (dict): Chứa dải snr_range_db, mảng detection_rates, detection_rates_strict, rmse_values, rmse_ms,
                conditional_rmse, conditional_rmse_ms, crb_samples, crb_ms, mean_errors, mean_errors_ms.
    """
    r_clean = create_received_signal(pulse, true_delay, sig_len)
    num_snr = len(snr_range_db)
    detection_rates = np.zeros(num_snr, dtype=float)
    detection_rates_strict = np.zeros(num_snr, dtype=float)
    rmse_values = np.zeros(num_snr, dtype=float)
    conditional_rmse = np.zeros(num_snr, dtype=float)
    mean_errors = np.zeros(num_snr, dtype=float)
    
    # Khởi tạo bộ sinh số ngẫu nhiên cố định seed để kết quả 100% tái lập được qua mọi lần chạy
    rng = np.random.RandomState(seed)
    
    if verbose:
        tol_ms = (tolerance / fs) * 1000.0
        print(f"\n[Monte Carlo] Chạy {num_snr} mức SNR ({snr_range_db[0]}dB -> {snr_range_db[-1]}dB) x {n_trials} lần thử (Seed={seed}, Dung sai ≤ {tol_ms:.1f}ms / {tolerance} mẫu)...")
        
    # Vòng lặp quét qua từng giá trị SNR trong dải kiểm tra
    for i, snr in enumerate(snr_range_db):
        trial_errors = []
        # Chạy n_trials lần thử ngẫu nhiên với các biến thể nhiễu
        for trial in range(n_trials):
            r_noisy = add_awgn(r_clean, snr, peak_amplitude=peak_amp, rng=rng)
            d_hat, _, _ = estimate_delay(pulse, r_noisy)
            err = compute_error(d_hat, true_delay)
            trial_errors.append(err)
            
        # Tổng hợp các chỉ số đánh giá cho mức SNR hiện tại
        detection_rates[i] = calculate_detection_rate(trial_errors, tolerance)
        detection_rates_strict[i] = calculate_detection_rate(trial_errors, tolerance_strict)
        rmse_values[i] = calculate_rmse(trial_errors)
        
        # Tính RMSE có điều kiện (chỉ xét các lần phát hiện thành công trong dung sai <= tolerance)
        cond_errors = [e for e in trial_errors if e <= tolerance]
        conditional_rmse[i] = calculate_rmse(cond_errors) if len(cond_errors) > 0 else 0.0
        
        mean_errors[i] = calculate_mean_error(trial_errors)
        
        # In thông báo tiến độ định kỳ mỗi 5 dB
        if verbose and (int(snr) % 5 == 0 or i == 0 or i == num_snr - 1):
            rmse_ms_val = (rmse_values[i] / fs) * 1000.0
            cond_rmse_ms_val = (conditional_rmse[i] / fs) * 1000.0
            print(f"  SNR = {snr:+3.0f} dB | Detection (≤2ms) = {detection_rates[i]:5.1f}% | RMSE = {rmse_values[i]:6.3f} mẫu ({rmse_ms_val:5.3f} ms) | RMSE (Locked) = {conditional_rmse[i]:5.3f} mẫu ({cond_rmse_ms_val:5.3f} ms)")
            
    if verbose:
        print("[Monte Carlo] Hoàn tất!")
        
    rmse_ms = (rmse_values / fs) * 1000.0
    conditional_rmse_ms = (conditional_rmse / fs) * 1000.0
    mean_errors_ms = (mean_errors / fs) * 1000.0
    
    # Tính toán đường giới hạn lý thuyết CRB động cho xung phát
    crb_samples = compute_cramer_rao_bound(pulse, snr_range_db, peak_amp=peak_amp)
    crb_ms = (crb_samples / fs) * 1000.0
    
    return {
        "snr_range_db": snr_range_db,
        "detection_rates": detection_rates,
        "detection_rates_strict": detection_rates_strict,
        "rmse_values": rmse_values,
        "rmse_ms": rmse_ms,
        "conditional_rmse": conditional_rmse,
        "conditional_rmse_ms": conditional_rmse_ms,
        "crb_samples": crb_samples,
        "crb_ms": crb_ms,
        "mean_errors": mean_errors,
        "mean_errors_ms": mean_errors_ms,
        "tolerance_samples": tolerance,
        "tolerance_ms": (tolerance / fs) * 1000.0,
        "tolerance_strict_samples": tolerance_strict,
        "fs": fs
    }

def save_metrics_to_csv(filepath: str,
                        mc_results: dict) -> None:
    """
    Chức năng: Ghi toàn bộ dữ liệu chỉ số mô phỏng Monte Carlo ra tệp CSV.
    Đầu vào:
      - filepath (str): Đường dẫn tệp CSV đầu ra.
      - mc_results (dict): Dictionary kết quả từ run_monte_carlo.
    Đầu ra:
      - None.
    """
    snr_range_db = mc_results["snr_range_db"]
    det_rates = mc_results["detection_rates"]
    det_strict = mc_results.get("detection_rates_strict", det_rates)
    rmse_samples = mc_results["rmse_values"]
    rmse_ms = mc_results["rmse_ms"]
    cond_rmse_samples = mc_results.get("conditional_rmse", rmse_samples)
    cond_rmse_ms = mc_results.get("conditional_rmse_ms", rmse_ms)
    mae_samples = mc_results["mean_errors"]
    mae_ms = mc_results["mean_errors_ms"]
    
    with open(filepath, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "SNR_dB",
            "Detection_Rate_leq_2ms_pct",
            "Detection_Rate_leq_2samples_pct",
            "RMSE_All_samples",
            "RMSE_All_ms",
            "RMSE_Locked_leq_2ms_samples",
            "RMSE_Locked_leq_2ms_ms",
            "MAE_samples",
            "MAE_ms"
        ])
        for i in range(len(snr_range_db)):
            writer.writerow([
                f"{snr_range_db[i]:.2f}",
                f"{det_rates[i]:.2f}",
                f"{det_strict[i]:.2f}",
                f"{rmse_samples[i]:.4f}",
                f"{rmse_ms[i]:.4f}",
                f"{cond_rmse_samples[i]:.4f}",
                f"{cond_rmse_ms[i]:.4f}",
                f"{mae_samples[i]:.4f}",
                f"{mae_ms[i]:.4f}"
            ])
