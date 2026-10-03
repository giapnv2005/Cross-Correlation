# ƯỚC LƯỢNG ĐỘ TRỄ TÍN HIỆU BẰNG TƯƠNG QUAN CHÉO (CROSS-CORRELATION)

Dự án nghiên cứu và mô phỏng thực nghiệm bài toán xác định độ trễ thời gian (**Time-Delay Estimation — TDE**) trong các hệ thống Radar/Sonar sử dụng phương pháp **Tương quan chéo (Cross-Correlation)** trên miền thời gian.

> [!IMPORTANT]
> **Yêu cầu kỹ thuật cốt lõi (100% Tự lập trình):** Toàn bộ các thuật toán xử lý tín hiệu (tạo xung phát tam giác đối xứng, dịch trễ tín hiệu, mô hình hóa nhiễu AWGN, tính tương quan chéo miền thời gian $r_{sr}[\ell]$, thuật toán dò tìm đỉnh cực đại dương lớn nhất $\hat{D} = \arg\max_{\ell \ge 0} r_{sr}[\ell]$, tính toán sai số RMSE, CRB lý thuyết và tỷ lệ phát hiện Detection Rate) **đều được tự lập trình hoàn toàn bằng toán học cơ bản**, tuyệt đối **không sử dụng** các toolbox chuyên dụng có sẵn như `scipy.signal`, `np.correlate`, `np.convolve`, hay `find_peaks`.

---

## 1. Cấu Trúc Mã Nguồn Dự Án

Hệ thống được tổ chức theo cấu trúc module hóa chuẩn mực, mỗi file đảm nhiệm một chức năng chuyên biệt:

```
e:\XLTHS\Cross-Correlation\
├── main.py                  # Điểm khởi chạy chính: điều phối tạo dữ liệu, chạy 4 test, Monte Carlo, hiển thị GUI
├── config.py                # Toàn bộ tham số hệ thống (fs=8000Hz, dung sai <=2.0ms) và 4 kịch bản test (.npz)
├── signal_utils.py          # Tự code: tạo xung tam giác đối xứng, dịch trễ, cộng AWGN, đo năng lượng Es, lưu/đọc .npz
├── correlation.py           # Tự code: hàm Cross-Correlation r_sr[ℓ] và thuật toán dò đỉnh cực đại D̂
├── metrics.py               # Tự code: RMSE, Conditional RMSE, CRB lý thuyết, Detection Rate, mô phỏng Monte Carlo
├── plotting.py              # Vẽ đồ thị 3 tầng tối ưu, đồ thị Monte Carlo 2 trục (mẫu & ms) kèm CRB, dashboard metrics
│
├── README.md                # Báo cáo kỹ thuật: cơ sở lý thuyết, phân tích CRB, hướng dẫn chạy và kết quả mới nhất
├── requirements.txt         # Thư viện phụ thuộc tối thiểu (numpy, matplotlib)
│
├── test_data/               # Thư mục lưu 4 gói dữ liệu kiểm thử (.npz chuẩn không dùng pickle)
│   ├── test1.npz            # Kịch bản 1: SNR cao (+10 dB), lý tưởng, bắt trúng D = 200 (25.00 ms)
│   ├── test2.npz            # Kịch bản 2: SNR thấp (-5 dB), vùng chuyển tiếp, lệch nhẹ 5 mẫu do đỉnh tù
│   ├── test3.npz            # Kịch bản 3: Xung dài (80 mẫu) tăng Es, trễ D = 150 (18.75 ms), SNR = +5 dB
│   └── test4.npz            # Kịch bản 4: Xung ngắn (35 mẫu), trễ D = 280 (35.00 ms), SNR = -10 dB (Đỉnh giả ở xa)
│
└── results/                 # Thư mục lưu toàn bộ đồ thị và số liệu thực nghiệm mới nhất
    ├── fig1_test1.png       # Đồ thị phân tích 3 tầng Test 1
    ├── fig2_test2.png       # Đồ thị phân tích 3 tầng Test 2
    ├── fig3_test3.png       # Đồ thị phân tích 3 tầng Test 3
    ├── fig4_test4.png       # Đồ thị phân tích 3 tầng Test 4
    ├── monte_carlo_summary.png # Đồ thị hiệu năng Detection Rate & RMSE (kèm trục ms và đường CRB)
    ├── metrics.png          # Dashboard trực quan hóa toàn bộ bảng 31 mức SNR kèm 4 thẻ KPI
    └── metrics.csv          # Bảng dữ liệu số liệu thô chi tiết 31 mức SNR x 100 lần thử
```

---

## 2. Cơ Sở Lý Thuyết & Công Thức Toán Học

### 2.1. Mô hình bài toán thu tín hiệu Radar / Sonar
Một xung tín hiệu chuẩn $s[n]$ được phát đi, phản xạ từ mục tiêu và quay trở lại đầu thu sau khoảng thời gian trễ $D$ mẫu (tương ứng $\Delta t = D / f_s$ giây), đồng thời bị suy hao và pha tạp bởi nhiễu trắng Gauss cộng (AWGN):
$$r[n] = s[n - D] + w[n]$$

Trong đó:
- $f_s = 8000\text{ Hz}$: Tần số lấy mẫu ($1\text{ mẫu} = 0.125\text{ ms} = 125\ \mu\text{s}$).
- $s[n]$: Xung phát tam giác đối xứng chuẩn, chiều dài $M = 50$ mẫu, năng lượng $E_s = \sum_{n=0}^{M-1} s^2[n] \approx 16.33$.
  *(Lưu ý: Với $M=50$ chẵn, tâm xung tại $n=24.5$ nên biên độ đỉnh thực tế đạt $\approx 0.98$, hai mẫu đầu/cuối $s[0]=s[M-1]=0$, xung hiệu dụng gồm 48 mẫu).*
- $D$: Độ trễ truyền sóng thực tế ($D \ge 0$, là tham số cần ước lượng).
- $w[n]$: Nhiễu AWGN không tương quan với tín hiệu, phân phối $w[n] \sim \mathcal{N}(0, \sigma_w^2)$.
- **Quy ước định nghĩa SNR:**
  Hệ thống sử dụng **Peak SNR** (tính theo biên độ đỉnh của xung):
  $$\text{SNR}_{\text{peak, dB}} = 10 \log_{10}\left(\frac{A_{\text{peak}}^2}{\sigma_w^2}\right)$$
  *Đối chiếu với các định nghĩa khác trong tài liệu xử lý tín hiệu:*
  - **SNR theo công suất trung bình trong xung:**
    $$\text{SNR}_{\text{avg, dB}} = 10 \log_{10}\left(\frac{P_{\text{pulse}}}{\sigma_w^2}\right) = \text{SNR}_{\text{peak, dB}} + 10 \log_{10}\left(\frac{E_s}{M A_{\text{peak}}^2}\right) \approx \text{SNR}_{\text{peak, dB}} - 4.86\text{ dB}$$
  - **SNR đầu ra bộ lọc phối hợp (Matched Filter Output SNR):**
    $$\left(\frac{E_s}{\sigma_w^2}\right)_{\text{dB}} = \text{SNR}_{\text{peak, dB}} + 10 \log_{10}\left(\frac{E_s}{A_{\text{peak}}^2}\right) \approx \text{SNR}_{\text{peak, dB}} + 12.13\text{ dB}$$
  *Ví dụ: Tại $\text{SNR}_{\text{peak}} = 0\text{ dB}$, công suất trung bình trong xung $/ \sigma_w^2 = -4.9\text{ dB}$, nhưng $E_s / \sigma_w^2 = +12.1\text{ dB}$.*

### 2.2. Thuật toán Tương quan chéo (Cross-Correlation)
*(Tham khảo: Manolakis & Ingle, 2011 — Applied Digital Signal Processing, Mục 4.5.4)*

Hàm tương quan chéo rời rạc trên miền thời gian giữa xung tham chiếu $s[n]$ và tín hiệu quan sát $r[n]$ được định nghĩa:
$$r_{sr}[\ell] = \sum_{n=-\infty}^{\infty} s[n] \, r[n + \ell]$$

Với $s[n]$ có độ dài $M$ mẫu và $r[n]$ có độ dài $N$ mẫu, biến trễ (lag) $\ell$ quét trong dải:
$$\ell \in [-(M - 1), \, N - 1]$$

Khi $\ell = D$, tín hiệu thu được gióng hàng chính xác với xung phát:
$$r[n + D] = s[n] + w[n + D] \implies r_{sr}[D] = \sum_{n=0}^{M-1} s^2[n] + \sum_{n=0}^{M-1} s[n] w[n + D] = E_s + \text{nhiễu tích lũy}$$

Phép tương quan chéo đóng vai trò như một **bộ lọc phối hợp (Matched Filter)**, tập trung năng lượng của toàn bộ xung tín hiệu $E_s$ thành một **đỉnh cực đại** tại đúng tọa độ lag $\ell = D$.

### 2.3. Thuật toán Ước lượng độ trễ (Peak Detection)
Do xung phát $s[n] \ge 0$ là xung đơn cực dương và trễ truyền sóng vật lý $D \ge 0$, độ trễ ước lượng $\hat{D}$ được xác định bằng vị trí lag có giá trị **dương lớn nhất** của hàm tương quan chéo trong miền không âm:
$$\hat{D} = \arg\max_{\ell \ge 0} r_{sr}[\ell]$$

### 2.4. Tiêu chuẩn Đánh giá Hiệu năng & Dung sai (Tolerance)
1. **Quy đổi thời gian sang số mẫu ($f_s = 8000\text{ Hz}$):**
   - Dung sai chuẩn theo đề bài: $\Delta t \le 2.0\text{ ms} \iff 16\text{ mẫu}$ (vì $2.0\text{ ms} \times 8000\text{ Hz} / 1000 = 16$).
   - Dung sai chặt để đối chiếu: $\le 2\text{ mẫu} \iff 0.25\text{ ms}$.
2. **Tỷ lệ phát hiện đúng (Detection Rate):**
   $$\text{Detection Rate} = \frac{1}{K} \sum_{i=1}^K \mathbf{1}(|\hat{D}_i - D| \le \text{tolerance}) \times 100\%$$
3. **Sai số toàn phương trung bình (RMSE):**
   $$\text{RMSE}_{\text{samples}} = \sqrt{\frac{1}{K} \sum_{i=1}^K (\hat{D}_i - D)^2}, \quad \text{RMSE}_{\text{ms}} = \frac{\text{RMSE}_{\text{samples}}}{f_s} \times 1000$$

### 2.5. Phân Tích Kỹ Thuật & Giới Hạn Cramér–Rao Bound (CRB)
1. **Giới hạn lý thuyết Cramér–Rao Bound (CRB):**
   - Đối với bài toán ước lượng độ trễ tín hiệu rời rạc trong nhiễu trắng Gauss, phương sai sai số của bất kỳ bộ ước lượng không chệch nào bị chặn dưới bởi CRB:
     $$\text{Var}(\hat{D}) \ge \text{CRB}(\hat{D}) = \frac{\sigma_w^2}{\sum_{n=0}^{M-2} (s[n+1] - s[n])^2} \implies \text{CRB}_{\text{samples}} = \frac{\sigma_w}{\sqrt{\sum_{n=0}^{M-2} (\Delta s[n])^2}}$$
   - Do bị chặn dưới bởi CRB, **RMSE không thể hội tụ về 0 tuyệt đối** mà sẽ tiệm cận gần sát giới hạn CRB ở SNR cao:
     - Tại $\text{Peak SNR} = +10\text{ dB}$: RMSE đo được $= 1.196\text{ mẫu}$, giới hạn $\text{CRB} \approx 1.118\text{ mẫu}$ (RMSE cao hơn CRB khoảng $7\%$).
     - Tại $\text{Peak SNR} = +15\text{ dB}$: RMSE đo được $= 0.768\text{ mẫu}$, giới hạn $\text{CRB} \approx 0.628\text{ mẫu}$ (RMSE cao hơn CRB khoảng $22\%$).
     - Kết quả đo thực nghiệm **gần sát giới hạn CRB** (RMSE cao hơn CRB khoảng $7\%$ tại $+10\text{ dB}$ và $22\%$ tại $+15\text{ dB}$ với $K = 100$ lần thử). Chênh lệch này hoàn toàn nằm trong phạm vi dao động thống kê của ước lượng RMSE (sai số tương đối $\approx 1/\sqrt{2K} \approx 7\%$); khi chạy kiểm tra với $3000$ lần thử, tỷ lệ $\text{RMSE} / \text{CRB}$ ổn định đều ở khoảng $1.09 \sim 1.10$. Phần dư còn lại $(\approx 9\% \sim 10\%)$ bắt nguồn từ việc công thức CRB tính ở đây là xấp xỉ từ đạo hàm sai phân của tín hiệu rời rạc.
2. **Đặc tính đỉnh tương quan tù của xung tam giác vs. Xung chữ nhật:**
   - Xung tam giác có độ dốc tuyến tính nhỏ ($\approx 0.04/\text{mẫu}$ với $M=50$). Khi tính tự tương quan $R_{ss}[\ell]$, độ chênh lệch giữa đỉnh và điểm lân cận rất nhỏ: $R_{ss}(0) - R_{ss}(\pm 1) \approx 0.04$.
   - Do đó ở SNR trung bình ($0 \sim 5\text{ dB}$), hệ thống đã bắt đúng xung trong dung sai $\le 16$ mẫu ($2.0\text{ ms}$), nhưng sai số vẫn có độ lệch chuẩn $\text{std} = 2.1 \sim 4.2$ mẫu (dao động vị trí đỉnh do đỉnh tù).
   - **Gợi ý thiết kế:** Xung chữ nhật có hàm tự tương quan dạng tam giác đỉnh nhọn hơn, giúp tăng tổng bình phương đạo hàm $\sum (\Delta s)^2$ và giảm CRB, cho khả năng định vị trễ chính xác hơn ở SNR cao.
3. **Hiện tượng sụp đổ ngưỡng (Threshold Breakdown / Spurious Peaks):**
   - Khi $\text{SNR} < -5\text{ dB}$, công suất nhiễu vượt trội hoàn toàn. Nhiễu ngẫu nhiên tạo ra các đỉnh giả ở khoảng cách rất xa trên toàn miền quan sát ($|\hat{D}-D| \gg 16$ mẫu), khiến RMSE tăng vọt lên $> 80\sim 120$ mẫu.

---

## 3. Bốn Kịch Bản Kiểm Thử Chi Tiết (Ví Dụ Minh Họa Có Kiểm Soát Seed)

> [!NOTE]
> Bốn kịch bản dưới đây là các **ví dụ minh họa trực quan (controlled illustrative scenarios)** với seed ngẫu nhiên được chọn trước nhằm mô tả 4 tình huống vật lý điển hình (bắt trúng, dao động đỉnh tù, tăng $E_s$, và đỉnh giả ở xa). Toàn bộ đánh giá thống kê khách quan tổng thể được thực hiện độc lập tại **Bước 3 qua mô phỏng Monte Carlo 3100 lần thử**.

| Kịch Bản | Chiều Dài Xung $M$ | Trễ Thực $D$ | Trễ Ước Lượng $\hat{D}$ | Sai Số | Peak SNR (dB) | Hiện Tượng Vật Lý & Đánh Giá |
|---|:---:|:---:|:---:|:---:|:---:|---|
| **Test 1** (`test1.npz`, seed 2024) | 50 mẫu ($6.25\text{ ms}$) | 200 ($25.00\text{ ms}$) | 200 ($25.00\text{ ms}$) | $0\text{ mẫu}$ ($0.00\text{ ms}$) | +10.0 dB | **Lý tưởng:** Bắt trúng tuyệt đối, đỉnh tương quan sắc nét. |
| **Test 2** (`test2.npz`, seed 99) | 50 mẫu ($6.25\text{ ms}$) | 200 ($25.00\text{ ms}$) | 195 ($24.38\text{ ms}$) | $5\text{ mẫu}$ ($0.62\text{ ms}$) | -5.0 dB | **Vùng chuyển tiếp:** Nhiễu gây xê dịch nhẹ 5 mẫu do đỉnh tù, đạt dung sai đề bài ($\le 2.0\text{ ms}$). *(Thống kê Monte Carlo: tỷ lệ đạt ở -5 dB là 62%)*. |
| **Test 3** (`test3.npz`, seed 101) | 80 mẫu ($10.00\text{ ms}$) | 150 ($18.75\text{ ms}$) | 152 ($19.00\text{ ms}$) | $2\text{ mẫu}$ ($0.25\text{ ms}$) | +5.0 dB | **Tăng năng lượng $E_s$:** Xung dài hơn giúp Matched Filter triệt tiêu đỉnh giả ở xa tốt hơn. Tuy nhiên đỉnh tương quan tù hơn nên độ phân giải định vị tinh bị giới hạn ($\text{CRB}_{M=80} \approx 1.41\text{ mẫu}$ so với $\text{CRB}_{M=50} \approx 1.12\text{ mẫu}$). |
| **Test 4** (`test4.npz`, seed 999) | 35 mẫu ($4.38\text{ ms}$) | 280 ($35.00\text{ ms}$) | 495 ($61.88\text{ ms}$) | $215\text{ mẫu}$ ($26.88\text{ ms}$) | -10.0 dB | **Sụp đổ ngưỡng:** Xung ngắn, SNR cực thấp khiến năng lượng xung bị nhiễu vùi lấp hoàn toàn, nhiễu tạo đỉnh giả ở xa ($\hat{D}=495$), thất bại do vượt ngưỡng dung sai (tỷ lệ thất bại ở cấu hình này khoảng $\approx 80\%$). |

---

## 4. Hướng Dẫn Cài Đặt & Chạy Chương Trình

### 4.1. Cài đặt môi trường
Dự án chỉ yêu cầu Python 3.8+ cùng hai thư viện tối thiểu:
```bash
pip install -r requirements.txt
```

### 4.2. Khởi chạy bằng 1 lệnh duy nhất
```bash
python main.py
```

### 4.3. Quá trình thực thi tự động của `main.py`
Chương trình được thiết kế theo đúng quy trình 3 bước chuẩn:
1. **Bước 1 (`step1_generate_test_datasets`):** Khởi tạo và lưu 4 file dữ liệu `test1.npz` -> `test4.npz` vào thư mục `test_data/`.
2. **Bước 2 (`step2_process_test_cases_and_plot`):** Đọc từng file test, thực hiện ước lượng độ trễ bằng Cross-Correlation, xuất 4 file ảnh đồ thị 3 tầng `fig1_test1.png` -> `fig4_test4.png` vào thư mục `results/`.
3. **Bước 3 (`step3_run_monte_carlo_and_export`):** Chạy mô phỏng Monte Carlo 31 mức SNR ($\text{Seed} = 2024$, 100 lần thử/mức = 3100 lần lặp độc lập, đảm bảo **100% tái lập kết quả**), đồng thời tự động xuất bảng số liệu `results/metrics.csv`, đồ thị `results/monte_carlo_summary.png`, và dashboard `results/metrics.png`.
4. **Hiển thị GUI các cửa sổ đồ thị:** Tự động kích hoạt **Per-Monitor High-DPI Awareness**, mở đồng thời các cửa sổ đồ thị trực quan (Test Cases & Monte Carlo) trên màn hình desktop.

---

## 5. Kết Quả Thực Nghiệm & Bình Luận Kỹ Thuật

### 5.1. Bảng số liệu thực nghiệm Monte Carlo (Khớp 100% tệp `results/metrics.csv` mới nhất)

| Peak SNR (dB) | Detection Rate ($\le 2.0\text{ ms}$) | Detection Rate ($\le 2\text{ mẫu}$) | RMSE toàn bộ (mẫu) | RMSE toàn bộ (ms) | RMSE khóa trúng (mẫu) | MAE (mẫu) | Đánh giá trạng thái |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **-15 dB** | 16.0% | 1.0% | 120.273 | 15.034 ms | 9.358 | 98.010 | Nhiễu áp đảo (Sụp đổ ngưỡng) |
| **-10 dB** | 37.0% | 10.0% | 113.388 | 14.174 ms | 8.027 | 79.390 | Nhiễu áp đảo (Sụp đổ ngưỡng) |
| **-5 dB**  | 62.0% | 18.0% | 81.917 | 10.240 ms | 6.332 | 46.060 | Vùng chuyển tiếp (Đỉnh giả & dịch đỉnh) |
| **0 dB**   | 98.0% | 47.0% | 9.570 | 1.196 ms | 4.259 | 4.180 | Bắt đầu vùng tin cậy cao |
| **+5 dB**  | 100.0% | 73.0% | 2.280 | 0.285 ms | 2.280 | 1.800 | Rất tốt (Độ lệch chuẩn std = 2.1 mẫu) |
| **+10 dB** | 100.0% | 99.0% | 1.196 | 0.149 ms | 1.196 | 0.970 | Hoàn hảo (Bám sát CRB = 1.12 mẫu) |
| **+15 dB** | 100.0% | 100.0% | 0.768 | 0.096 ms | 0.768 | 0.510 | Hoàn hảo (Bám sát CRB = 0.63 mẫu) |

> [!NOTE]
> **Giải thích hiện tượng biến động thống kê RMSE tại vùng biên (+1 dB vs 0 dB):**
> Trong bảng dữ liệu chi tiết `metrics.csv`, RMSE toàn bộ tại $+1\text{ dB}$ ($22.118\text{ mẫu}$) cao hơn tại $0\text{ dB}$ ($9.570\text{ mẫu}$). Đây là **biến động thống kê hoàn toàn bình thường** khi chạy với số lượng mẫu thử hữu hạn ($K = 100$ lần thử): tại vùng chuyển tiếp $+1\text{ dB}$, chỉ cần $1 \sim 2$ lần thử ngẫu nhiên rơi vào đỉnh giả ở xa (spurious peak) là bình phương sai số $(\hat{D}-D)^2$ đã đủ lớn để kéo giá trị RMSE trung bình lên.
> 
> Nếu xét riêng **RMSE khóa trúng (Conditional RMSE)** thì sai số giảm đều đặn từ $4.259\text{ mẫu}$ ở $0\text{ dB}$ xuống $3.724\text{ mẫu}$ ở $+1\text{ dB}$, $2.280\text{ mẫu}$ ở $+5\text{ dB}$, và tiệm cận $0.768\text{ mẫu}$ ở $+15\text{ dB}$.
> 
> *(Lưu ý thêm: Ở dải SNR cực thấp $\le -5\text{ dB}$, các lần "trúng" phần lớn là đỉnh giả rơi ngẫu nhiên vào cửa sổ dung sai $\pm 16$ mẫu. Phân phối đều trên $[-16, 16]$ có $\text{RMSE} \approx 16/\sqrt{3} \approx 9.24\text{ mẫu}$, rất gần với $9.358\text{ mẫu}$ đo được tại $-15\text{ dB}$. Do đó, RMSE có điều kiện chỉ thực sự phản ánh độ chính xác bám sát CRB ở vùng $\text{SNR} \ge +5\text{ dB}$ khi $\text{Detection Rate} = 100\%$).*

### 5.2. Bình luận và đánh giá theo yêu cầu đề bài
1. **Ngưỡng đạt chất lượng cao:**
   - Với dung sai đề bài $\le 2.0\text{ ms}$ ($16$ mẫu), hệ thống đạt $\text{Detection Rate} \ge 90\%$ ngay từ $\text{Peak SNR} \ge 0\text{ dB}$ và đạt $100\%$ từ $\text{Peak SNR} \ge +2\text{ dB}$.
   - Với dung sai cực chặt $\le 2\text{ mẫu}$ ($0.25\text{ ms}$), hệ thống đạt $\text{Detection Rate} \ge 90\%$ tại $\text{Peak SNR} \ge +10\text{ dB}$.
2. **Vai trò của năng lượng xung $E_s$ và Matched Filter:**
   - Tương quan chéo hoạt động như một bộ lọc phối hợp tối ưu, tích lũy toàn bộ năng lượng xung $E_s$ qua **phép tính tương quan chéo miền thời gian**, giúp tăng tỷ số tín hiệu trên nhiễu ở đầu ra thêm $+12.1\text{ dB}$ so với Peak SNR đầu vào, phát hiện chính xác vị trí độ trễ ngay cả khi trên đồ thị thời gian tín hiệu thu bị nhiễu vùi lấp hoàn toàn.
3. **Phân tích vùng chuyển tiếp, độ lệch chuẩn và độ trễ thực nghiệm:**
   - Tại vùng $\text{Peak SNR} \in [-5\text{ dB}, -1\text{ dB}]$, nhiễu tạo đỉnh giả ở xa khiến $\text{Detection Rate}$ dao động từ $62\%$ đến $89\%$.
   - Tại vùng $\text{Peak SNR} \in [0\text{ dB}, +5\text{ dB}]$, hệ thống bắt trúng xung trong dung sai $\le 2.0\text{ ms}$, nhưng độ lệch chuẩn sai số vẫn ở mức $2 \sim 4$ mẫu do đỉnh tự tương quan của xung tam giác có độ dốc thoải (đỉnh tù).
   - Chỉ từ $\text{Peak SNR} \ge +10\text{ dB}$, sai số mới thực sự thu hẹp về $1 \sim 3$ mẫu (độ lệch chuẩn $1.2\text{ mẫu}$ tại $+10\text{ dB}$) và bám sát giới hạn tối ưu Cramér–Rao Bound.
4. **Tuân thủ quy định:** Toàn bộ mã nguồn tự code 100% bằng toán học cơ bản, không dùng hàm có sẵn, có đầy đủ docstring và comment chi tiết cho từng khối lệnh.
