import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import numpy as np

def get_screen_work_area():
    """
    Chức năng: Lấy tọa độ và kích thước vùng làm việc màn hình (loại trừ thanh Taskbar trên Windows).
    Đầu vào: Không có.
    Đầu ra: (offset_x, offset_y, width, height) hoặc None nếu không dùng được Windows API.
    """
    try:
        import ctypes
        import ctypes.wintypes
        try:
            ctypes.windll.shcore.SetProcessDpiAwareness(2)
        except Exception:
            try:
                ctypes.windll.user32.SetProcessDPIAware()
            except Exception:
                pass
                
        rect = ctypes.wintypes.RECT()
        SPI_GETWORKAREA = 0x0030
        if ctypes.windll.user32.SystemParametersInfoW(SPI_GETWORKAREA, 0, ctypes.byref(rect), 0):
            w = rect.right - rect.left
            h = rect.bottom - rect.top
            return rect.left, rect.top, w, h
    except Exception:
        pass
    return None

def place_window_corner(fig=None, position: str = "top-right"):
    """
    Chức năng: Căn chỉnh kích thước và vị trí cửa sổ đồ thị để 4 cửa sổ phủ kín toàn bộ màn hình desktop.
    Đầu vào:
      - fig (matplotlib.figure.Figure, tùy chọn): Đối tượng Figure cần định vị.
      - position (str): Vị trí góc ("top-left", "top-right", "bottom-left", "bottom-right").
    Đầu ra:
      - None.
    """
    if fig is None:
        fig = plt.gcf()
        
    backend = matplotlib.get_backend().lower()
    if 'agg' in backend and backend != 'tkagg':
        print(f"[Cảnh báo GUI] Backend hiện tại là '{backend}' (không tương tác), không thể điều khiển cửa sổ đồ thị.")
        return

    try:
        manager = fig.canvas.manager
        if manager is None:
            return

        # Tính toán tọa độ và kích thước cửa sổ đối với backend Tkinter (TkAgg)
        if hasattr(manager, "window") and hasattr(manager.window, "wm_geometry"):
            root = manager.window
            root.update_idletasks()
            
            # Lấy vùng làm việc thực tế từ Windows API để phủ kín toàn bộ diện tích desktop
            work_area = get_screen_work_area()
            if work_area is not None:
                offset_x, offset_y, screen_w, screen_h = work_area
            else:
                offset_x, offset_y = 0, 0
                screen_w = root.winfo_screenwidth()
                screen_h = root.winfo_screenheight()
            
            # Chia màn hình thành 4 ô góc bằng nhau phủ kín 100% desktop (mỗi ô là 1/4 màn hình)
            w_half = screen_w // 2
            h_half = screen_h // 2
            
            # Phân bổ vị trí x, y theo 4 góc phần tư màn hình
            if position == "top-left":
                x, y = offset_x, offset_y
            elif position == "top-right":
                x, y = offset_x + w_half, offset_y
            elif position == "bottom-left":
                x, y = offset_x, offset_y + h_half
            elif position == "bottom-right":
                x, y = offset_x + w_half, offset_y + h_half
            else:
                x, y = offset_x, offset_y
                
            # Đặt kích thước và tọa độ cho cửa sổ Tkinter để phủ kín toàn màn hình
            win_w = w_half - 16
            win_h = h_half - 45
            root.wm_geometry(f"{int(win_w)}x{int(win_h)}+{int(x)}+{int(y)}")

            # Trên Windows: gọi MoveWindow trên HWND để định vị chuẩn xác tuyệt đối từng pixel bao gồm cả viền
            try:
                import ctypes
                hwnd = ctypes.windll.user32.GetParent(root.winfo_id())
                if not hwnd:
                    hwnd = root.winfo_id()
                if hwnd:
                    ctypes.windll.user32.MoveWindow(hwnd, int(x), int(y), int(w_half), int(h_half), True)
            except Exception:
                pass

        # Đặt tọa độ vị trí cửa sổ đối với backend PyQt hoặc PySide (QtAgg)
        elif hasattr(manager, "window") and hasattr(manager.window, "setGeometry"):
            work_area = get_screen_work_area()
            sw = work_area[2] if work_area else 1920
            sh = work_area[3] if work_area else 1020
            ww, wh = sw // 2, sh // 2
            if position == "top-left":
                x, y = 0, 0
            elif position == "top-right":
                x, y = ww, 0
            elif position == "bottom-left":
                x, y = 0, wh
            elif position == "bottom-right":
                x, y = ww, wh
            else:
                x, y = 0, 0
            manager.window.setGeometry(int(x), int(y), int(ww), int(wh))
    except Exception as e:
        print(f"[Cảnh báo GUI] Gặp lỗi khi căn chỉnh vị trí cửa sổ '{position}': {e}")

def plot_test_case_figure(test_id: str,
                          test_name: str,
                          s: np.ndarray,
                          r_noisy: np.ndarray,
                          corr: np.ndarray,
                          lags: np.ndarray,
                          true_delay: int,
                          est_delay: int,
                          snr_db: float,
                          fs: float = 8000.0,
                          tolerance: int = 16,
                          save_path: str = None,
                          corner: str = "top-right") -> plt.Figure:
    """
    Chức năng: Vẽ đồ thị phân tích chi tiết 3 tầng cho một kịch bản kiểm thử:
               (a) Xung phát s[n], (b) Tín hiệu thu r[n], (c) Hàm tương quan chéo r_sr[ℓ].
    Đầu vào:
      - test_id (str): Mã định danh kịch bản (ví dụ: 'test1').
      - test_name (str): Tên mô tả của kịch bản kiểm thử.
      - s (np.ndarray): Mảng mẫu xung phát s[n].
      - r_noisy (np.ndarray): Tín hiệu thu kèm nhiễu r[n].
      - corr (np.ndarray): Mảng hàm tương quan chéo r_sr[ℓ].
      - lags (np.ndarray): Mảng chỉ số độ lệch lag ℓ.
      - true_delay (int): Độ trễ thực tế D (mẫu).
      - est_delay (int): Độ trễ ước lượng D̂ (mẫu).
      - snr_db (float): Mức SNR của kịch bản (dB).
      - fs (float): Tần số lấy mẫu hệ thống (Hz).
      - tolerance (int): Dung sai bắt trúng (mẫu, mặc định: 16 mẫu = 2.0 ms).
      - save_path (str, tùy chọn): Đường dẫn lưu tệp hình ảnh PNG.
      - corner (str): Vị trí góc màn hình ("top-left", "top-right", "bottom-left", "bottom-right").
    Đầu ra:
      - fig (matplotlib.figure.Figure): Đối tượng đồ thị đã khởi tạo và vẽ hoàn chỉnh.
    """
    # Sử dụng height_ratios để mở rộng panel (c) hàm tương quan chéo rộng rãi hơn
    fig, axes = plt.subplots(3, 1, figsize=(9.6, 5.4), dpi=100, gridspec_kw={'height_ratios': [1.0, 1.0, 1.45]})
    fig.patch.set_facecolor('#FAFAFA')

    # Đánh giá sai số ước lượng theo cả số mẫu và thời gian thực ms
    error_samples = abs(est_delay - true_delay)
    error_ms = (error_samples / fs) * 1000.0
    true_ms = (true_delay / fs) * 1000.0
    est_ms = (est_delay / fs) * 1000.0
    
    is_hit = error_samples <= tolerance
    status_text = "✓ ĐẠT (≤ 2.0 ms)" if is_hit else "✗ THẤT BẠI (> 2.0 ms)"
    status_color = "#2E7D32" if is_hit else "#C62828"

    # Hiển thị tiêu đề súc tích, rõ ràng kèm cả đơn vị mẫu và ms
    fig.suptitle(f"{test_id.upper()}: {test_name}\n"
                 f"SNR = {snr_db:+.1f} dB | Trễ thực D = {true_delay} ({true_ms:.2f} ms) | "
                 f"Ước lượng $\\hat{{D}}$ = {est_delay} ({est_ms:.2f} ms) | Sai số: {error_samples} mẫu ({error_ms:.2f} ms) [{status_text}]",
                 fontsize=9.2, fontweight='bold', color='#1A237E', y=0.985)

    # Tầng 1: Đồ thị xung phát chuẩn s[n] trên miền thời gian (đầy đủ title và axis labels)
    ax1 = axes[0]
    ax1.set_facecolor('#FFFFFF')
    n_s = np.arange(len(s))
    ax1.plot(n_s, s, color='#0D47A1', linewidth=1.8, label=f's[n] (M={len(s)})')
    ax1.fill_between(n_s, s, alpha=0.18, color='#1976D2')
    ax1.set_xlabel('Chỉ số mẫu n', fontsize=8, fontweight='bold')
    ax1.set_ylabel('Biên độ s[n]', fontsize=8, fontweight='bold')
    ax1.set_title('(a) Tín hiệu phát chuẩn s[n] (Xung tam giác đối xứng)', fontsize=8.5, fontweight='bold', loc='left', pad=2)
    ax1.grid(True, linestyle='--', alpha=0.5)
    ax1.legend(loc='upper right', fontsize=7.5, framealpha=0.85)

    # Tầng 2: Đồ thị tín hiệu quan sát r[n] có chứa nhiễu AWGN và đánh dấu vùng trễ thực
    ax2 = axes[1]
    ax2.set_facecolor('#FFFFFF')
    n_r = np.arange(len(r_noisy))
    ax2.plot(n_r, r_noisy, color='#546E7A', linewidth=0.9, alpha=0.85, label='r[n]')
    ax2.axvspan(true_delay, min(true_delay + len(s), len(r_noisy)), color='#4CAF50', alpha=0.22,
                label=f'Xung thực [{true_delay}, {true_delay + len(s)}]')
    ax2.set_xlabel('Chỉ số mẫu n', fontsize=8, fontweight='bold')
    ax2.set_ylabel('Biên độ r[n]', fontsize=8, fontweight='bold')
    ax2.set_title(f'(b) Tín hiệu thu r[n] = s[n - D] + w[n] (SNR = {snr_db:+.1f} dB)', fontsize=8.5, fontweight='bold', loc='left', pad=2)
    ax2.grid(True, linestyle='--', alpha=0.5)
    ax2.legend(loc='upper right', fontsize=7.5, framealpha=0.85)

    # Tầng 3: Đồ thị hàm tương quan chéo r_sr[ℓ], đỉnh ước lượng và vị trí trễ thực D
    ax3 = axes[2]
    ax3.set_facecolor('#FFFFFF')
    ax3.plot(lags, corr, color='#D84315', linewidth=1.4, label='$r_{sr}[\\ell]$')
    idx_est = np.where(lags == est_delay)[0]
    peak_val = corr[idx_est[0]] if len(idx_est) > 0 else np.max(corr)
    ax3.axvline(x=true_delay, color='#2E7D32', linestyle='--', linewidth=1.5, alpha=0.85, label=f'Trễ thực D={true_delay}')
    ax3.plot(est_delay, peak_val, 'o', color=status_color, markersize=6.5, markeredgecolor='black',
             markeredgewidth=1.2, label=f'Đỉnh $\\hat{{D}}$={est_delay} (Lag {est_delay})', zorder=5)
    ax3.set_xlabel('Độ lệch trễ lag $\\ell$ (mẫu)', fontsize=8.5, fontweight='bold')
    ax3.set_ylabel('$r_{sr}[\\ell]$', fontsize=8.5, fontweight='bold')
    ax3.set_title('(c) Kết quả Tương quan chéo và Vị trí đỉnh ước lượng độ trễ $\\hat{D}$', fontsize=8.5, fontweight='bold', loc='left', pad=2)
    ax3.set_xlim(min(lags), max(lags))
    ax3.margins(y=0.20)  # Thêm lề trục Y để đỉnh không bị cắt sát mép trên
    ax3.grid(True, linestyle='--', alpha=0.5)
    
    # Đặt vị trí legend linh hoạt để không bao giờ che mất đỉnh ước lượng
    lag_mid = (min(lags) + max(lags)) / 2.0
    legend_loc = 'upper left' if est_delay > lag_mid else 'upper right'
    ax3.legend(loc=legend_loc, fontsize=7.5, framealpha=0.9)

    # Canh lề bố cục vừa vặn không bị đè chữ
    plt.tight_layout(rect=[0.01, 0.01, 0.99, 0.95])
    place_window_corner(fig, position=corner)
    if save_path:
        plt.savefig(save_path, dpi=130, bbox_inches='tight')
    return fig

def plot_monte_carlo_results(mc_results: dict,
                             save_path: str = None) -> plt.Figure:
    """
    Chức năng: Vẽ 2 đồ thị hiệu năng Monte Carlo: Tỷ lệ phát hiện đúng (%) và sai số RMSE theo SNR (Thang Log).
               Có tích hợp trục phụ hiển thị sai số theo đơn vị mili-giây (ms) và đường lý thuyết CRB ở SNR cao.
    Đầu vào:
      - mc_results (dict): Dictionary kết quả từ run_monte_carlo.
      - save_path (str, tùy chọn): Đường dẫn tệp lưu ảnh đồ thị PNG.
    Đầu ra:
      - fig (matplotlib.figure.Figure): Đối tượng đồ thị đã vẽ.
    """
    snr_range_db = mc_results["snr_range_db"]
    det_rates = mc_results["detection_rates"]
    det_strict = mc_results.get("detection_rates_strict", det_rates)
    rmse_values = mc_results["rmse_values"]
    cond_rmse = mc_results.get("conditional_rmse", rmse_values)
    crb_samples = mc_results.get("crb_samples", None)
    tol_ms = mc_results.get("tolerance_ms", 2.0)
    tol_samples = mc_results.get("tolerance_samples", 16)
    fs = mc_results.get("fs", 8000.0)
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 8.8), dpi=120)
    fig.patch.set_facecolor('#FAFAFA')
    fig.suptitle('HIỆU NĂNG ƯỚC LƯỢNG ĐỘ TRỄ BẰNG CROSS-CORRELATION (MONTE CARLO)\n'
                 'Mô phỏng 31 mức SNR x 100 lần thử | fs = 8000 Hz | 100% Thuật toán tự code',
                 fontsize=12.5, fontweight='bold', color='#1A237E', y=0.98)

    # Đồ thị 1: Tỷ lệ phát hiện đúng (%) theo SNR (kèm cả 2 ngưỡng dung sai ≤ 2ms và ≤ 2 mẫu)
    ax1.set_facecolor('#FFFFFF')
    ax1.plot(snr_range_db, det_rates, 'o-', color='#1565C0', linewidth=2.2,
             markersize=5, markerfacecolor='white', markeredgewidth=1.8,
             label=f'Detection Rate (Dung sai đề bài ≤ {tol_ms:.1f} ms / {tol_samples} mẫu)')
    ax1.plot(snr_range_db, det_strict, 's--', color='#7B1FA2', linewidth=1.6,
             markersize=4, alpha=0.85, label='Detection Rate (Dung sai chặt ≤ 2 mẫu / 0.25 ms)')
    ax1.axhline(y=90, color='#2E7D32', linestyle='--', linewidth=1.5, alpha=0.8, label='Ngưỡng chuẩn 90%')
    ax1.axhline(y=50, color='#F57C00', linestyle=':', linewidth=1.5, alpha=0.8, label='Ngưỡng 50%')
    ax1.set_xlabel('Tỷ số Tín hiệu trên Nhiễu Peak SNR (dB)', fontsize=10, fontweight='bold')
    ax1.set_ylabel('Tỷ lệ phát hiện (%)', fontsize=10, fontweight='bold')
    ax1.set_ylim(-5, 105)
    ax1.set_xlim(snr_range_db[0] - 0.5, snr_range_db[-1] + 0.5)
    ax1.grid(True, linestyle='--', alpha=0.6)
    ax1.xaxis.set_major_locator(MultipleLocator(5))
    ax1.xaxis.set_minor_locator(MultipleLocator(1))
    ax1.legend(loc='lower right', framealpha=0.9, fontsize=9)
    ax1.set_title('(a) Tỷ lệ phát hiện đúng chính xác độ trễ (%) theo SNR', fontsize=10, fontweight='bold', loc='left')

    # Đồ thị 2: Sai số toàn phương trung bình RMSE theo SNR trên THANG LOGARITHM
    ax2.set_facecolor('#FFFFFF')
    ax2.set_yscale('log')
    
    line_rmse = ax2.plot(snr_range_db, rmse_values, 's-', color='#C62828', linewidth=2.2,
                         markersize=5, markerfacecolor='white', markeredgewidth=1.8, label='RMSE toàn bộ (số mẫu)')
    line_cond = ax2.plot(snr_range_db, cond_rmse, '^-.', color='#E65100', linewidth=1.6,
                         markersize=4.5, alpha=0.9, label='RMSE có điều kiện (chỉ tính lần trong dung sai ≤ 2.0 ms)')
    
    lines = line_rmse + line_cond
    
    # Chỉ vẽ đường CRB lý thuyết ở dải SNR >= 0 dB (vùng tin cậy cao trước khi sụp đổ ngưỡng)
    if crb_samples is not None:
        snr_crb_mask = snr_range_db >= 0
        line_crb = ax2.plot(snr_range_db[snr_crb_mask], crb_samples[snr_crb_mask], 'o:', color='#2E7D32', linewidth=2.0,
                            markersize=4, label='Giới hạn lý thuyết Cramér–Rao CRB (vùng SNR ≥ 0 dB)')
        lines = lines + line_crb
        
    line_tol = ax2.axhline(y=float(tol_samples), color='#1565C0', linestyle='--', linewidth=1.4, alpha=0.75, label=f'Ngưỡng dung sai đề bài {tol_samples} mẫu ({tol_ms:.1f} ms)')
    line_strict = ax2.axhline(y=2.0, color='#7B1FA2', linestyle=':', linewidth=1.4, alpha=0.75, label='Ngưỡng dung sai chặt 2 mẫu (0.25 ms)')
    lines = lines + [line_tol, line_strict]
    
    ax2.set_xlabel('Tỷ số Tín hiệu trên Nhiễu Peak SNR (dB)', fontsize=10, fontweight='bold')
    ax2.set_ylabel('RMSE (mẫu) [Thang Log]', fontsize=10, fontweight='bold', color='#C62828')
    ax2.set_xlim(snr_range_db[0] - 0.5, snr_range_db[-1] + 0.5)
    ax2.grid(True, which='both', linestyle='--', alpha=0.5)
    ax2.xaxis.set_major_locator(MultipleLocator(5))
    ax2.xaxis.set_minor_locator(MultipleLocator(1))
    ax2.tick_params(axis='y', labelcolor='#C62828')
    
    # Thiết lập trục phụ bên phải theo đơn vị thời gian mili-giây (ms) trên thang LOG đồng bộ
    ax2_right = ax2.twinx()
    ax2_right.set_yscale('log')
    y_min, y_max = ax2.get_ylim()
    ax2_right.set_ylim((y_min / fs) * 1000.0, (y_max / fs) * 1000.0)
    ax2_right.set_ylabel('RMSE (mili-giây ms) [Thang Log]', fontsize=10, fontweight='bold', color='#1A237E')
    ax2_right.tick_params(axis='y', labelcolor='#1A237E')
    
    # Tổng hợp legend cho đồ thị 2
    labels = [l.get_label() for l in lines]
    ax2.legend(lines, labels, loc='upper right', framealpha=0.9, fontsize=8.2)
    ax2.set_title('(b) Căn bậc hai sai số toàn phương trung bình (RMSE) theo SNR (Thang Log — Thấy rõ khoảng cách CRB)', fontsize=10, fontweight='bold', loc='left')

    plt.tight_layout(rect=[0, 0.02, 1, 0.95])
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    return fig

def plot_metrics_table_image(mc_results: dict,
                             save_path: str = None) -> plt.Figure:
    """
    Chức năng: Trực quan hóa toàn bộ bảng số liệu Monte Carlo dưới dạng dashboard hình ảnh
               với 4 thẻ KPI tóm tắt và bảng 2 cột phân loại màu sắc trạng thái.
    Đầu vào:
      - mc_results (dict): Dictionary kết quả từ run_monte_carlo.
      - save_path (str, tùy chọn): Đường dẫn tệp lưu ảnh dashboard PNG.
    Đầu ra:
      - fig (matplotlib.figure.Figure): Đối tượng Figure dashboard đã hoàn tất.
    """
    snr_range_db = mc_results["snr_range_db"]
    detection_rates = mc_results["detection_rates"]
    det_strict = mc_results.get("detection_rates_strict", detection_rates)
    rmse_values = mc_results["rmse_values"]
    rmse_ms = mc_results["rmse_ms"]
    
    fig = plt.figure(figsize=(15, 11), dpi=150)
    fig.patch.set_facecolor('#F8F9FA')

    # Hiển thị tiêu đề chính của dashboard và cấu hình bài toán
    fig.text(0.5, 0.965, 'BẢNG CHỈ SỐ ĐO HIỆU NĂNG MONTE CARLO — CROSS-CORRELATION',
             ha='center', va='center', fontsize=16, fontweight='bold', color='#1A237E')
    fig.text(0.5, 0.938, 'Mô hình: r[n] = s[n - D] + w[n] | fs = 8000 Hz | 31 mức SNR (-15 dB → +15 dB) × 100 lần thử | Dung sai ≤ 2.0 ms (16 mẫu)',
             ha='center', va='center', fontsize=10.5, color='#455A64')

    # Trích xuất các giá trị mốc quan trọng để điền vào 4 thẻ KPI tóm tắt
    idx_90 = np.where(detection_rates >= 90.0)[0]
    snr_90_str = f"≥ {snr_range_db[idx_90[0]]:+.0f} dB" if len(idx_90) > 0 else "> 15 dB"
    idx_100 = np.where(detection_rates >= 100.0)[0]
    snr_100_str = f"≥ {snr_range_db[idx_100[0]]:+.0f} dB" if len(idx_100) > 0 else "> 15 dB"
    idx_0 = np.where(snr_range_db == 0)[0]
    rmse_0 = rmse_values[idx_0[0]] if len(idx_0) > 0 else 0.0
    rmse_0_ms = rmse_ms[idx_0[0]] if len(idx_0) > 0 else 0.0
    idx_15 = np.where(snr_range_db == 15)[0]
    rmse_15 = rmse_values[idx_15[0]] if len(idx_15) > 0 else rmse_values[-1]
    rmse_15_ms = rmse_ms[idx_15[0]] if len(idx_15) > 0 else rmse_ms[-1]

    # Danh sách 4 thẻ KPI với màu sắc và nội dung tương ứng (Sửa: Tiệm cận giới hạn CRB)
    kpi_cards = [
        {"title": "Ngưỡng Detection ≥ 90% (≤2ms)", "value": snr_90_str, "sub": "Vùng tin cậy cao", "color": "#1B5E20", "bg": "#E8F5E9"},
        {"title": "Ngưỡng Bắt Trúng Tuyệt Đối 100%", "value": snr_100_str, "sub": "Xác suất bắt trúng 100%", "color": "#004D40", "bg": "#E0F2F1"},
        {"title": "Sai Số RMSE tại SNR = 0 dB", "value": f"{rmse_0:.2f} mẫu ({rmse_0_ms:.2f}ms)", "sub": "Vùng chuyển tiếp", "color": "#E65100", "bg": "#FFF3E0"},
        {"title": "Sai Số RMSE tại SNR = +15 dB", "value": f"{rmse_15:.3f} mẫu ({rmse_15_ms:.3f}ms)", "sub": "Tiệm cận giới hạn CRB", "color": "#311B92", "bg": "#EDE7F6"},
    ]

    # Vẽ các hình chữ nhật và nội dung văn bản cho 4 thẻ KPI
    card_y, card_h, card_w, spacing, start_x = 0.84, 0.065, 0.21, 0.02, 0.05
    for idx, card in enumerate(kpi_cards):
        cx = start_x + idx * (card_w + spacing)
        rect = plt.Rectangle((cx, card_y), card_w, card_h, transform=fig.transFigure,
                             facecolor=card['bg'], edgecolor=card['color'], linewidth=1.5,
                             clip_on=False, zorder=2)
        fig.patches.append(rect)
        fig.text(cx + card_w/2, card_y + card_h * 0.72, card['title'],
                 ha='center', va='center', fontsize=9.0, fontweight='bold', color=card['color'])
        fig.text(cx + card_w/2, card_y + card_h * 0.38, card['value'],
                 ha='center', va='center', fontsize=12.0, fontweight='bold', color=card['color'])
        fig.text(cx + card_w/2, card_y + card_h * 0.12, card['sub'],
                 ha='center', va='center', fontsize=8, color='#546E7A')

    # Hàm quy đổi tỷ lệ phát hiện và sai số thành nhãn trạng thái và màu nền ô
    def get_status_and_color(det, rmse):
        if det >= 99.0:
            return "Hoàn hảo", "#C8E6C9"
        elif det >= 90.0:
            return "Rất tốt", "#E8F5E9"
        elif det >= 70.0:
            return "Khá", "#E1F5FE"
        elif det >= 40.0:
            return "Chuyển tiếp", "#FFF3E0"
        else:
            return "Nhiễu áp đảo", "#FFEBEE"

    # Chuẩn bị dữ liệu cho bảng bên trái (các mức SNR từ -15 dB đến 0 dB)
    mid_idx = 16
    left_snrs, left_dets = snr_range_db[:mid_idx], detection_rates[:mid_idx]
    left_dets_strict = det_strict[:mid_idx]
    left_rmses, left_rmse_ms = rmse_values[:mid_idx], rmse_ms[:mid_idx]
    headers = ['SNR (dB)', 'Det (≤2ms)', 'Det (≤2 mẫu)', 'RMSE (mẫu)', 'RMSE (ms)', 'Đánh giá']

    ax_left = fig.add_axes([0.05, 0.08, 0.43, 0.73])
    ax_left.axis('off')
    table_left_data, cell_colors_left = [], []
    for i in range(len(left_snrs)):
        status, bg_color = get_status_and_color(left_dets[i], left_rmses[i])
        table_left_data.append([
            f"{left_snrs[i]:+.0f} dB",
            f"{left_dets[i]:5.1f}%",
            f"{left_dets_strict[i]:5.1f}%",
            f"{left_rmses[i]:6.2f}",
            f"{left_rmse_ms[i]:5.2f}",
            status
        ])
        cell_colors_left.append(["#F5F5F5" if i % 2 == 0 else "#FFFFFF"] * 5 + [bg_color])

    # Khởi tạo bảng bên trái và định dạng màu chữ tiêu đề cột
    tbl_left = ax_left.table(cellText=table_left_data, colLabels=headers, cellColours=cell_colors_left,
                             colColours=['#1A237E']*6, cellLoc='center', loc='center')
    tbl_left.auto_set_font_size(False)
    tbl_left.set_fontsize(8.2)
    tbl_left.scale(1.0, 1.45)
    for col in range(6):
        tbl_left[(0, col)].get_text().set_color('white')
        tbl_left[(0, col)].get_text().set_fontweight('bold')

    # Chuẩn bị dữ liệu cho bảng bên phải (các mức SNR từ +1 dB đến +15 dB)
    right_snrs, right_dets = snr_range_db[mid_idx:], detection_rates[mid_idx:]
    right_dets_strict = det_strict[mid_idx:]
    right_rmses, right_rmse_ms = rmse_values[mid_idx:], rmse_ms[mid_idx:]

    ax_right = fig.add_axes([0.52, 0.08, 0.43, 0.73])
    ax_right.axis('off')
    table_right_data, cell_colors_right = [], []
    for i in range(len(right_snrs)):
        status, bg_color = get_status_and_color(right_dets[i], right_rmses[i])
        table_right_data.append([
            f"{right_snrs[i]:+.0f} dB",
            f"{right_dets[i]:5.1f}%",
            f"{right_dets_strict[i]:5.1f}%",
            f"{right_rmses[i]:6.2f}",
            f"{right_rmse_ms[i]:5.2f}",
            status
        ])
        cell_colors_right.append(["#F5F5F5" if i % 2 == 0 else "#FFFFFF"] * 5 + [bg_color])

    # Cân bằng số dòng giữa hai bảng bằng một dòng đệm rỗng
    if len(right_snrs) < len(left_snrs):
        table_right_data.append(["-", "-", "-", "-", "-", "-"])
        cell_colors_right.append(["#FAFAFA"] * 6)

    # Khởi tạo bảng bên phải và định dạng màu chữ tiêu đề cột
    tbl_right = ax_right.table(cellText=table_right_data, colLabels=headers, cellColours=cell_colors_right,
                               colColours=['#1A237E']*6, cellLoc='center', loc='center')
    tbl_right.auto_set_font_size(False)
    tbl_right.set_fontsize(8.2)
    tbl_right.scale(1.0, 1.45)
    for col in range(6):
        tbl_right[(0, col)].get_text().set_color('white')
        tbl_right[(0, col)].get_text().set_fontweight('bold')

    # Chú thích chân trang và lưu ảnh dashboard
    fig.text(0.5, 0.025,
             'Ghi chú: Dung sai quy ước chuẩn theo đề bài là |Δt̂ - Δt| ≤ 2.0 ms (16 mẫu tại fs=8000Hz). Cột dung sai chặt ≤ 2 mẫu (0.25ms) dùng để đối chiếu.',
             ha='center', va='center', fontsize=9, fontstyle='italic', color='#546E7A')
    if save_path:
        plt.savefig(save_path, dpi=160, bbox_inches='tight')

    return fig
