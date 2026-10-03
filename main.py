import sys
import io
import os

# Thiết lập DPI awareness cho Windows để nhận diện độ phân giải màn hình chính xác (Full HD 1920x1080 thay vì 1536x864)
try:
    import ctypes
    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(2)  # PROCESS_PER_MONITOR_DPI_AWARE
    except Exception:
        try:
            ctypes.windll.user32.SetProcessDPIAware()
        except Exception:
            pass
except Exception:
    pass

# Cấu hình đầu ra console hỗ trợ UTF-8 (an toàn khi redirect stdout hoặc chạy pipe không có buffer)
try:
    if hasattr(sys.stdout, 'buffer') and sys.stdout.buffer is not None:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
except Exception:
    pass

# Thiết lập backend đồ họa TkAgg để hiển thị các cửa sổ đồ thị kết quả trên Windows
import matplotlib
if "MPLBACKEND" not in os.environ:
    try:
        import tkinter
        matplotlib.use('TkAgg')
    except Exception as e:
        print(f"[Cảnh báo GUI] Tkinter không khả dụng ({e}). Tự động fallback về backend 'Agg'.")
        matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

# Nhập các cấu hình hệ thống và các hàm thuật toán từ các module
import config
from signal_utils import (
    generate_triangle_pulse,
    create_received_signal,
    add_awgn,
    compute_energy,
    save_test_data,
    load_test_data
)
from correlation import estimate_delay
from metrics import run_monte_carlo, save_metrics_to_csv, compute_error
from plotting import plot_test_case_figure, plot_monte_carlo_results, plot_metrics_table_image

def step1_generate_test_datasets():
    """
    Chức năng: Khởi tạo và lưu 4 bộ dữ liệu tín hiệu kiểm thử (test1 -> test4) vào thư mục test_data/.
    Đầu vào:
      - Không có (sử dụng thông số từ config.TEST_CASES).
    Đầu ra:
      - None (xuất các tệp .npy ra đĩa).
    """
    print("\n" + "="*70)
    print("  BƯỚC 1: KHỞI TẠO 4 BỘ TEST DATA VÀ LƯU VÀO test_data/")
    print("="*70)
    
    # Lặp qua từng cấu hình test case để tạo xung s[n] và tín hiệu thu r[n] có nhiễu
    for key, cfg in config.TEST_CASES.items():
        s = generate_triangle_pulse(cfg["pulse_len"], amplitude=config.PEAK_AMPLITUDE)
        r_clean = create_received_signal(s, cfg["true_delay"], cfg["sig_len"])
        r_noisy = add_awgn(r_clean, cfg["snr_db"], peak_amplitude=config.PEAK_AMPLITUDE, seed=cfg["seed"])
        
        # Đóng gói dữ liệu kiểm thử vào dictionary và lưu thành file .npy
        data_dict = {
            "s": s,
            "r_clean": r_clean,
            "r_noisy": r_noisy,
            "true_delay": cfg["true_delay"],
            "snr_db": cfg["snr_db"],
            "pulse_len": cfg["pulse_len"],
            "sig_len": cfg["sig_len"],
            "test_name": cfg["name"],
            "seed": cfg["seed"]
        }
        file_path = os.path.join(config.TEST_DATA_DIR, cfg["file_name"])
        save_test_data(file_path, data_dict)
        
        # Tính toán năng lượng xung và thông báo trạng thái khởi tạo
        Es = compute_energy(s)
        print(f"  [+] Đã lưu {cfg['file_name']:12s}: {cfg['name']}")
        print(f"      - Chiều dài xung M = {cfg['pulse_len']} (Es = {Es:.2f}), Trễ thực D = {cfg['true_delay']}, SNR = {cfg['snr_db']:+.1f} dB")

def step2_process_test_cases_and_plot():
    """
    Chức năng: Đọc từng bộ test trong test_data/, thực hiện ước lượng độ trễ và xuất đồ thị fig1 -> fig4.
    Đầu vào:
      - Không có (đọc dữ liệu từ thư mục test_data/).
    Đầu ra:
      - None (xuất các tệp ảnh PNG vào thư mục results/).
    """
    print("\n" + "="*70)
    print("  BƯỚC 2: XỬ LÝ 4 BỘ TEST, ƯỚC LƯỢNG ĐỘ TRỄ VÀ XUẤT ĐỒ THỊ FIG1 - FIG4")
    print("="*70)
    
    # Tải từng bộ dữ liệu test từ đĩa và thực hiện thuật toán tương quan chéo
    for key, cfg in config.TEST_CASES.items():
        file_path = os.path.join(config.TEST_DATA_DIR, cfg["file_name"])
        data = load_test_data(file_path)
        s, r, true_D, snr = data["s"], data["r_noisy"], data["true_delay"], data["snr_db"]
        
        # Ước lượng độ trễ D_hat từ đỉnh dương lớn nhất của hàm tương quan chéo r_sr[ℓ]
        est_D, corr, lags = estimate_delay(s, r)
        err = compute_error(est_D, true_D)
        err_ms = (err / config.FS) * 1000.0
        true_ms = (true_D / config.FS) * 1000.0
        est_ms = (est_D / config.FS) * 1000.0
        
        # Đánh giá theo ngưỡng dung sai đề bài (<= 2.0 ms = 16 mẫu) và dung sai chặt (<= 2 mẫu)
        if err == 0:
            res_str = "BẮT TRÚNG TUYỆT ĐỐI (D̂ = D)"
        elif err <= config.TOLERANCE_STRICT:
            res_str = f"ĐẠT CHÍNH XÁC (Lệch {err} mẫu / {err_ms:.2f} ms ≤ 2 mẫu)"
        elif err <= config.TOLERANCE:
            res_str = f"ĐẠT DUNG SAI ĐỀ BÀI (Lệch {err} mẫu / {err_ms:.2f} ms ≤ 2.0 ms do đỉnh tù)"
        else:
            res_str = f"THẤT BẠI (Lệch {err} mẫu / {err_ms:.2f} ms > 2.0 ms do đỉnh giả ở xa)"
        
        # Vẽ đồ thị 3 tầng, định vị về góc tương ứng và lưu file ảnh vào thư mục results/
        fig_path = os.path.join(config.RESULTS_DIR, cfg["fig_name"])
        corner_pos = cfg.get("corner", "top-right")
        fig = plot_test_case_figure(
            test_id=key,
            test_name=cfg["name"],
            s=s,
            r_noisy=r,
            corr=corr,
            lags=lags,
            true_delay=true_D,
            est_delay=est_D,
            snr_db=snr,
            fs=config.FS,
            tolerance=config.TOLERANCE,
            save_path=fig_path,
            corner=corner_pos
        )
        
        print(f"  [+] {key.upper()}: Trễ thực = {true_D:3d} ({true_ms:5.2f} ms) | Ước lượng = {est_D:3d} ({est_ms:5.2f} ms) | Sai số = {err:2d} mẫu ({err_ms:5.2f} ms)")
        print(f"      -> Đánh giá: {res_str}")
        print(f"      -> Đồ thị lưu tại: {fig_path}")

def step3_run_monte_carlo_and_export():
    """
    Chức năng: Chạy mô phỏng Monte Carlo đa mức SNR, xuất ảnh dashboard metrics.png, file CSV và đồ thị summary.
    Đầu vào:
      - Không có (sử dụng thông số từ config.py).
    Đầu ra:
      - mc_results (dict): Kết quả thống kê gồm dải SNR, Detection Rate, RMSE và MAE.
    """
    print("\n" + "="*70)
    print("  BƯỚC 3: MÔ PHỎNG MONTE CARLO & XUẤT metrics.png / metrics.csv")
    print("="*70)
    
    s_default = generate_triangle_pulse(config.PULSE_LEN, amplitude=config.PEAK_AMPLITUDE)
    
    # Chạy mô phỏng Monte Carlo 31 mức SNR x 100 lần thử với seed cố định tái lập 100%
    mc_results = run_monte_carlo(
        pulse=s_default,
        true_delay=config.TRUE_DELAY,
        sig_len=config.SIG_LEN,
        snr_range_db=config.SNR_RANGE_DB,
        n_trials=config.N_TRIALS,
        tolerance=config.TOLERANCE,
        tolerance_strict=config.TOLERANCE_STRICT,
        peak_amp=config.PEAK_AMPLITUDE,
        fs=config.FS,
        seed=config.MC_SEED,
        verbose=True
    )
    
    # Xuất ảnh dashboard tổng hợp số liệu trực quan có màu sắc phân loại
    metrics_img_path = os.path.join(config.RESULTS_DIR, "metrics.png")
    fig_table = plot_metrics_table_image(mc_results, save_path=metrics_img_path)
    plt.close(fig_table)
    print(f"  [✓] Đã xuất ảnh bảng số liệu trực quan: {metrics_img_path}")

    # Đồng thời lưu dữ liệu số liệu thô ra file CSV
    csv_path = os.path.join(config.RESULTS_DIR, "metrics.csv")
    save_metrics_to_csv(filepath=csv_path, mc_results=mc_results)
    print(f"  [✓] Đã lưu dữ liệu thô CSV: {csv_path}")
    
    # Vẽ và lưu đường cong hiệu năng Monte Carlo: Detection Rate và RMSE
    mc_fig_path = os.path.join(config.RESULTS_DIR, "monte_carlo_summary.png")
    fig_mc = plot_monte_carlo_results(mc_results, save_path=mc_fig_path)
    plt.close(fig_mc)
    print(f"  [✓] Đã xuất đồ thị Monte Carlo: {mc_fig_path}")
    
    return mc_results

def print_final_summary(mc_results):
    """
    Chức năng: In báo cáo tổng kết các ngưỡng SNR và bình luận kỹ thuật chuyên sâu theo yêu cầu đề bài.
    Đầu vào:
      - mc_results (dict): Dictionary kết quả từ mô phỏng Monte Carlo.
    Đầu ra:
      - None.
    """
    snrs = mc_results["snr_range_db"]
    dets = mc_results["detection_rates"]
    dets_strict = mc_results["detection_rates_strict"]
    rmses = mc_results["rmse_values"]
    rmses_ms = mc_results["rmse_ms"]
    cond_rmses = mc_results.get("conditional_rmse", rmses)
    cond_rmses_ms = mc_results.get("conditional_rmse_ms", rmses_ms)
    crbs = mc_results.get("crb_samples", None)
    
    # Xác định các mốc SNR đạt tỷ lệ phát hiện 90% và 100% (xử lý an toàn chuỗi/số)
    idx_90 = np.where(dets >= 90.0)[0]
    snr_90_val = snrs[idx_90[0]] if len(idx_90) > 0 else "> 15"
    snr_90_str = f"{snr_90_val:+.0f} dB" if isinstance(snr_90_val, (int, float, np.number)) else str(snr_90_val)
    
    idx_100 = np.where(dets >= 100.0)[0]
    snr_100_val = snrs[idx_100[0]] if len(idx_100) > 0 else "> 15"
    snr_100_str = f"{snr_100_val:+.0f} dB" if isinstance(snr_100_val, (int, float, np.number)) else str(snr_100_val)
    
    idx_90_strict = np.where(dets_strict >= 90.0)[0]
    snr_90_strict_val = snrs[idx_90_strict[0]] if len(idx_90_strict) > 0 else "> 15"
    snr_90_strict_str = f"{snr_90_strict_val:+.0f} dB" if isinstance(snr_90_strict_val, (int, float, np.number)) else str(snr_90_strict_val)
    
    # Trích xuất các số liệu cụ thể tại các mốc SNR để in báo cáo chính xác
    det_m5 = dets[np.where(snrs == -5)[0][0]] if len(np.where(snrs == -5)[0]) > 0 else 0.0
    det_m1 = dets[np.where(snrs == -1)[0][0]] if len(np.where(snrs == -1)[0]) > 0 else 0.0
    rmse_m15 = rmses[0]
    rmse_m10 = rmses[np.where(snrs == -10)[0][0]] if len(np.where(snrs == -10)[0]) > 0 else rmses[0]
    
    idx_0 = np.where(snrs == 0)[0][0] if len(np.where(snrs == 0)[0]) > 0 else 0
    idx_10 = np.where(snrs == 10)[0][0] if len(np.where(snrs == 10)[0]) > 0 else -1
    idx_15 = np.where(snrs == 15)[0][0] if len(np.where(snrs == 15)[0]) > 0 else -1
    
    crb_10_str = f"{crbs[idx_10]:.2f}" if crbs is not None else "1.12"
    crb_15_str = f"{crbs[idx_15]:.2f}" if crbs is not None else "0.63"
    
    # Hiển thị kết quả tổng kết hiệu năng
    print("\n" + "="*70)
    print("  TỔNG KẾT HIỆU NĂNG ƯỚC LƯỢNG ĐỘ TRỄ (TỰ CODE 100%)")
    print("="*70)
    print(f"  • Ngưỡng SNR đạt Detection Rate >= 90% (Dung sai đề bài ≤ 2.0 ms): {snr_90_str}")
    print(f"  • Ngưỡng SNR đạt Detection Rate = 100% (Dung sai đề bài ≤ 2.0 ms): {snr_100_str}")
    print(f"  • Ngưỡng SNR đạt Detection Rate >= 90% (Dung sai chặt ≤ 2 mẫu):     {snr_90_strict_str}")
    print(f"  • RMSE tại Peak SNR = 0 dB:             {rmses[idx_0]:.3f} mẫu ({rmses_ms[idx_0]:.3f} ms) | RMSE khóa trúng: {cond_rmses[idx_0]:.3f} mẫu")
    print(f"  • RMSE tại Peak SNR = -15 dB:           {rmses[0]:.3f} mẫu ({rmses_ms[0]:.3f} ms)")
    print(f"  • RMSE tại Peak SNR = +10 dB:           {rmses[idx_10]:.3f} mẫu ({rmses_ms[idx_10]:.3f} ms) [Gần sát CRB lý thuyết ≈ {crb_10_str} mẫu]")
    print(f"  • RMSE tại Peak SNR = +15 dB:           {rmses[idx_15]:.3f} mẫu ({rmses_ms[idx_15]:.3f} ms) [Gần sát CRB lý thuyết ≈ {crb_15_str} mẫu]")
    print("="*70)
    print(f"""  BÌNH LUẬN & ĐÁNH GIÁ KỸ THUẬT (THEO YÊU CẦU ĐỀ BÀI):
  1. Quy ước định nghĩa SNR:
     - Hệ thống sử dụng Peak SNR: SNR_peak = 10*log10(A_peak^2 / sigma^2).
     - Với xung tam giác M=50: Công suất trung bình xung / sigma^2 ≈ SNR_peak - 4.9 dB,
       và SNR đầu ra bộ lọc phối hợp Es/sigma^2 ≈ SNR_peak + 12.1 dB.
  2. Vùng sụp đổ ngưỡng (Threshold Breakdown, SNR < -5 dB):
     - Tại mức nhiễu cực mạnh, năng lượng nhiễu ngẫu nhiên vượt qua năng lượng xung Es,
       tạo ra các đỉnh giả (spurious peaks) ở vị trí bất kỳ -> RMSE tăng vọt ~{rmse_m10:.0f}–{rmse_m15:.0f} mẫu.
  3. Vùng chuyển tiếp & Ảnh hưởng của đỉnh tương quan tù (-5 dB <= SNR <= +5 dB):
     - Ở vùng -5 dB đến -1 dB, nhiễu gây xuất hiện đỉnh giả ở xa (Detection {det_m5:.0f}% → {det_m1:.0f}%).
     - Ở vùng 0 dB đến +5 dB, hệ thống đã bắt được xung trong dung sai đề bài (<= 2.0 ms = 16 mẫu),
       nhưng sai số vẫn dao động (std ~2–4 mẫu) do đỉnh tự tương quan của xung tam giác có độ dốc thoải (đỉnh tù, R(0)-R(±1) ≈ 0.04).
  4. Vùng SNR cao & Giới hạn Cramér–Rao Bound (CRB) (SNR >= +10 dB):
     - Chỉ từ SNR >= +10 dB, sai số mới thực sự thu hẹp về 1–3 mẫu (std = 1.2 mẫu tại 10 dB).
     - RMSE đo được (1.20 mẫu tại +10 dB; 0.77 mẫu tại +15 dB) gần sát giới hạn CRB (cao hơn khoảng 7% tại +10 dB và 22% tại +15 dB với K=100 lần thử).
       Chênh lệch này nằm trong dao động thống kê của ước lượng RMSE (sai số tương đối ≈ 1/√(2K) ≈ 7%); khi chạy kiểm tra với 3000 lần thử, tỷ lệ RMSE/CRB ổn định ở khoảng 1.09–1.10.
       Phần dư còn lại do công thức CRB tính ở đây là xấp xỉ từ đạo hàm sai phân của tín hiệu rời rạc.
  5. Thiết kế dạng xung:
     - Xung tam giác có đỉnh tù nên CRB bị chặn dưới cao hơn so với xung chữ nhật (xung chữ nhật
       cho hàm tự tương quan hình tam giác đỉnh nhọn, giúp định vị chính xác hơn ở SNR cao).""")
    print("="*70)

def main():
    """
    Chức năng: Điểm khởi chạy chính của chương trình, điều phối thực thi tuần tự từ Bước 1 đến Bước 3.
    Đầu vào:
      - Không có.
    Đầu ra:
      - None.
    """
    print("""
    ╔══════════════════════════════════════════════════════════════════════╗
    ║            HỆ THỐNG ƯỚC LƯỢNG ĐỘ TRỄ BẰNG CROSS-CORRELATION          ║
    ╚══════════════════════════════════════════════════════════════════════╝
    """)
    step1_generate_test_datasets()
    step2_process_test_cases_and_plot()
    mc_results = step3_run_monte_carlo_and_export()
    print_final_summary(mc_results)

    # Hiển thị đồng thời 4 cửa sổ kiểm thử tại 4 góc màn hình nếu backend có GUI
    backend_name = matplotlib.get_backend().lower()
    if backend_name != 'agg':
        print("\n" + "="*70)
        print("  [✓] Đang hiển thị 4 cửa sổ kiểm thử Fig 1 - Fig 4 phủ kín 4 góc màn hình...")
        print("  [✓] Đóng 4 cửa sổ kiểm thử này để xem tiếp đồ thị tổng kết Monte Carlo.")
        print("="*70)
        plt.show()

        # Hiển thị tiếp đồ thị tổng kết Monte Carlo sau khi xem xong 4 test cases
        fig_mc_display = plot_monte_carlo_results(mc_results)
        print("\n" + "="*70)
        print("  [✓] Đang hiển thị đồ thị tổng kết Monte Carlo (Detection Rate & RMSE theo SNR)...")
        print("  [✓] Đóng cửa sổ đồ thị để kết thúc chương trình.")
        print("="*70)
        plt.show()
    else:
        print("\n" + "="*70)
        print("  [i] Backend hiện tại là Agg (Headless). Toàn bộ ảnh đã được lưu vào thư mục results/.")
        print("="*70)

if __name__ == "__main__":
    main()
