import streamlit as st
import os
import sys
import subprocess

# ========== 处理 OpenCV 导入 ==========
try:
    import cv2
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "opencv-python-headless"])
    import cv2

# 导入其他库
try:
    from ultralytics import YOLO
except ImportError as e:
    st.error(f"❌ YOLO 导入失败: {e}")
    st.stop()

import csv
import io
import zipfile
import tempfile
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import make_interp_spline
from scipy import stats
import re
from pathlib import Path
import pandas as pd
from PIL import Image
import glob
import math

# ========== 页面配置 ==========
st.set_page_config(
    page_title="科学数据分析工具集",
    page_icon="🔬",
    layout="wide"
)

st.title("🔬 科学数据分析工具集")
st.markdown("选择左侧功能板块开始分析")

# ========== 侧边栏导航 ==========
st.sidebar.title("📋 功能导航")
page = st.sidebar.radio(
    "选择分析工具",
    ["📟 数码管数字识别", "⚡ B-Z振荡反应分析", "📊 整合CSV分析"],
    index=0
)

st.sidebar.markdown("---")
st.sidebar.caption("v1.0 | 支持批量数据处理")

# ============================================================
# B-Z分析核心函数
# ============================================================
def format_sigfigs(value, sigfigs=4):
    """Format a number with specified number of significant figures."""
    if value == 0 or np.isnan(value) or np.isinf(value):
        return f"{value:.4f}"
    if abs(value) < 0.0001 or abs(value) >= 10000:
        return f"{value:.{sigfigs-1}E}"
    if value < 0:
        value = -value
        neg = True
    else:
        neg = False
    if value >= 1:
        decimals = sigfigs - 1 - int(math.floor(math.log10(value)))
        if decimals < 0:
            decimals = 0
    else:
        decimals = sigfigs - 1
        while value * (10 ** decimals) < 10 ** (sigfigs - 1):
            decimals += 1
    if neg:
        value = -value
    return f"{value:.{decimals}f}"

def find_induction_min(potential, delta):
    """Determine the induction end point: first significant local minimum."""
    n = len(potential)
    min_val = potential[0]
    min_idx = 0
    i = 0
    while i < n - 1:
        if potential[i + 1] < min_val:
            min_val = potential[i + 1]
            min_idx = i + 1
            i += 1
        elif potential[i + 1] == min_val:
            i += 1
        else:
            peak = potential[i + 1]
            j = i + 1
            while j < n - 1 and potential[j + 1] >= potential[j]:
                peak = potential[j + 1]
                j += 1
            if peak - min_val >= delta:
                break
            else:
                i = j
                min_val = potential[i]
                min_idx = i
    return min_idx, min_val

def find_next_peak(time, potential, start_idx, delta):
    """Find the next significant local maximum."""
    n = len(potential)
    i = start_idx
    while i < n - 1:
        if potential[i + 1] <= potential[i]:
            i += 1
            continue
        best_idx = i + 1
        best_val = potential[i + 1]
        j = i + 1
        while j < n - 1:
            if potential[j + 1] > best_val:
                best_val = potential[j + 1]
                best_idx = j + 1
                j += 1
            elif potential[j + 1] == best_val:
                j += 1
            else:
                break
        if j >= n - 1:
            return best_idx, time[best_idx], best_val
        low = potential[j + 1]
        k = j + 1
        while k < n - 1 and potential[k + 1] <= potential[k]:
            low = potential[k + 1]
            k += 1
        if best_val - low >= delta:
            return best_idx, time[best_idx], best_val
        else:
            i = k
    idx = min(i, n - 1)
    return idx, time[idx], potential[idx]

def find_next_valley(time, potential, start_idx, delta):
    """Find the next significant local minimum."""
    n = len(potential)
    i = start_idx
    while i < n - 1:
        if potential[i + 1] >= potential[i]:
            i += 1
            continue
        best_idx = i + 1
        best_val = potential[i + 1]
        j = i + 1
        while j < n - 1:
            if potential[j + 1] < best_val:
                best_val = potential[j + 1]
                best_idx = j + 1
                j += 1
            elif potential[j + 1] == best_val:
                j += 1
            else:
                break
        if j >= n - 1:
            return best_idx, time[best_idx], best_val
        high = potential[j + 1]
        k = j + 1
        while k < n - 1 and potential[k + 1] >= potential[k]:
            high = potential[k + 1]
            k += 1
        if high - best_val >= delta:
            return best_idx, time[best_idx], best_val
        else:
            i = k
    idx = min(i, n - 1)
    return idx, time[idx], potential[idx]

def parse_temperature_from_file(filename):
    """Parse temperature from filename."""
    base = os.path.splitext(os.path.basename(filename))[0]
    m = re.search(r"-?\d+(?:\.\d+)?", base)
    if m:
        return float(m.group())
    return None

def extract_bz_from_dataframe(df, temp, pot_low, pot_high, n_valley, n_peak, delta):
    """Extract B-Z data from a DataFrame with time and potential columns."""
    # 获取时间和电势列
    time_raw = df.iloc[:, 0].values.astype(float)
    potential_raw = df.iloc[:, 1].values.astype(float)

    valid_mask = (potential_raw >= pot_low) & (potential_raw <= pot_high)
    time = time_raw[valid_mask]
    potential = potential_raw[valid_mask]
    
    if len(time) == 0:
        return None, [], [], None, [], []

    min_idx, min_val = find_induction_min(potential, delta)
    min_time = time[min_idx]
    induction_time = min_time

    induction_pts = []
    t = 0.0
    while t < min_time:
        idx = np.argmin(np.abs(time - t))
        induction_pts.append((time[idx], potential[idx]))
        t += 30.0
    induction_pts.append((min_time, min_val))

    osc_pts = [(min_time, min_val)]
    current_idx = min_idx + 1
    n_valley_count = 1
    n_peak_count = 0
    
    valley_times = [min_time]
    peak_times = []
    
    while n_valley_count < n_valley or n_peak_count < n_peak:
        if current_idx >= len(potential):
            break
        if n_peak_count < n_peak:
            idx, tp, pp = find_next_peak(time, potential, current_idx, delta)
            osc_pts.append((tp, pp))
            peak_times.append(tp)
            current_idx = idx + 1
            n_peak_count += 1
        if n_valley_count < n_valley and current_idx < len(potential):
            idx, tv, pv = find_next_valley(time, potential, current_idx, delta)
            osc_pts.append((tv, pv))
            valley_times.append(tv)
            current_idx = idx + 1
            n_valley_count += 1

    t_list = [p for p, _ in induction_pts[:-1]] + [p for p, _ in osc_pts]
    e_list = [v for _, v in induction_pts[:-1]] + [v for _, v in osc_pts]

    return temp, t_list, e_list, induction_time, valley_times, peak_times

def calculate_periods(valley_times, peak_times):
    """Calculate oscillation periods using valley and peak methods."""
    valley_periods = []
    if len(valley_times) >= 2:
        valley_periods = [valley_times[i+1] - valley_times[i] 
                          for i in range(len(valley_times)-1)]
    
    peak_periods = []
    if len(peak_times) >= 2:
        peak_periods = [peak_times[i+1] - peak_times[i] 
                        for i in range(len(peak_times)-1)]
    
    valley_period = np.mean(valley_periods) if valley_periods else np.nan
    peak_period = np.mean(peak_periods) if peak_periods else np.nan
    
    return valley_period, peak_period, valley_periods, peak_periods

def plot_time_series_single(data_dict):
    """Plot potential-time curves for each temperature."""
    fig_dict = {}
    for temp, (time_arr, pot_arr) in data_dict.items():
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(time_arr, pot_arr, 'b-o', markersize=6, markerfacecolor='b', 
               markeredgecolor='b', linewidth=1.5, label='Extracted data')
        ax.set_xlabel('Time / s')
        ax.set_ylabel('Potential / mV')
        ax.set_title(f'T = {temp:.1f} °C')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='best')
        plt.tight_layout()
        fig_dict[temp] = fig
        plt.close(fig)
    return fig_dict

def plot_arrhenius(induction_data, valley_period_data, peak_period_data):
    """Plot Arrhenius fitting plots."""
    def temp_to_kelvin(temp_c):
        return temp_c + 273.15
    
    temps_c = sorted(set(induction_data.keys()) | set(valley_period_data.keys()) | 
                     set(peak_period_data.keys()))
    
    if len(temps_c) < 2:
        return None, {}
    
    datasets = [
        ('Induction Period', induction_data, 'Induction'),
        ('Valley Method', valley_period_data, 'Valley'),
        ('Peak Method', peak_period_data, 'Peak')
    ]
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    colors = ['#2E86AB', '#A23B72', '#F18F01']
    
    results = {}
    R = 8.314
    
    for idx, (label, data_dict, key) in enumerate(datasets):
        ax = axes[idx]
        temps_available = [t for t in temps_c if t in data_dict and not np.isnan(data_dict[t])]
        
        if len(temps_available) < 2:
            ax.text(0.5, 0.5, 'Insufficient Data\n(Need at least 2 temperatures)', 
                   ha='center', va='center', transform=ax.transAxes, fontsize=14)
            ax.set_title(f'{label}')
            continue
        
        values = [data_dict[t] for t in temps_available]
        T_kelvin = [temp_to_kelvin(t) for t in temps_available]
        x_data = [1.0 / tk for tk in T_kelvin]
        y_data = np.log(1.0 / np.array(values))
        
        slope, intercept, r_value, p_value, std_err = stats.linregress(x_data, y_data)
        
        x_fit = np.linspace(min(x_data), max(x_data), 100)
        y_fit = slope * x_fit + intercept
        
        Ea = -slope * R
        Ea_kJ = Ea / 1000
        
        slope_str = format_sigfigs(slope, 4)
        intercept_str = format_sigfigs(intercept, 4)
        ea_str = format_sigfigs(Ea_kJ, 4)
        r2_str = format_sigfigs(r_value**2, 4)
        
        ax.scatter(x_data, y_data, color=colors[idx], s=80, zorder=5, 
                  label='Experimental data')
        ax.plot(x_fit, y_fit, color=colors[idx], linestyle='--', linewidth=2, 
               label=f'Fit: ln(1/t) = {slope_str}·(1/T) + {intercept_str}')
        
        ax.set_xlabel('1 / T (K⁻¹)')
        ax.set_ylabel('ln(1/t)')
        ax.set_title(f'{label}\nEa = {ea_str} kJ/mol, R² = {r2_str}')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='best', fontsize=9)
        
        results[key] = {
            'slope': slope,
            'intercept': intercept,
            'r_squared': r_value**2,
            'Ea_kJ': Ea_kJ,
            'equation': f'ln(1/t) = {slope_str}·(1/T) + {intercept_str}'
        }
    
    plt.tight_layout()
    return fig, results


# ============================================================
# 板块1：数码管数字识别
# ============================================================
def page_digital_tube():
    st.header("📟 数码管数字批量识别工具")
    st.markdown("上传包含数码管图片的 **ZIP 压缩包** 或 **视频文件**，系统将自动识别所有图片中的数字组合并生成 CSV 结果。")
    
    @st.cache_resource
    def load_model():
        model_path = "best.pt"
        if not os.path.exists(model_path):
            st.error(f"❌ 模型文件 '{model_path}' 未找到，请确保它位于项目根目录。")
            return None
        try:
            model = YOLO(model_path)
            return model
        except Exception as e:
            st.error(f"❌ 模型加载失败: {e}")
            return None

    model = load_model()
    if model is None:
        st.stop()

    with st.sidebar:
        st.header("⚙️ 参数设置")
        input_type = st.radio(
            "选择输入类型",
            ["📁 图片压缩包 (ZIP)", "🎬 视频文件"],
            index=0
        )
        
        fps_choice = None
        if input_type == "🎬 视频文件":
            st.subheader("🎞️ 抽帧设置")
            fps_choice = st.selectbox(
                "抽帧频率 (每秒帧数)",
                options=[0.5, 1, 2, 5, 10, 15, 30],
                index=1,
                format_func=lambda x: f"{x} 帧/秒" if x != 0.5 else "每2秒1帧"
            )
        
        save_images = st.checkbox("保存带检测框的结果图片", value=True)
        save_frames = st.checkbox("保存抽帧原图（仅视频模式）", value=True) if input_type == "🎬 视频文件" else False
        generate_plot = st.checkbox("生成电动势-时间平滑曲线图", value=True)
        conf_threshold = st.slider("置信度阈值", 0.0, 1.0, 0.25, 0.05)

    def extract_frame_number(filename):
        patterns = [
            r'(\d+)',
            r'frame[_\s-]?(\d+)',
            r'img[_\s-]?(\d+)',
            r'pic[_\s-]?(\d+)',
            r'f(\d+)',
            r'(\d{4})',
        ]
        for pattern in patterns:
            match = re.search(pattern, filename, re.IGNORECASE)
            if match:
                return int(match.group(1))
        return None

    def process_images(image_files, model, save_images, conf_threshold):
        results_data = []
        result_images = {}
        frame_images = {}
        progress_bar = st.progress(0, text="开始处理...")
        status_text = st.empty()

        for idx, (name, img_data) in enumerate(image_files.items()):
            status_text.text(f"正在处理 [{idx+1}/{len(image_files)}]: {name}")
            progress_bar.progress((idx + 1) / len(image_files))

            if isinstance(img_data, bytes):
                nparr = np.frombuffer(img_data, np.uint8)
                img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            else:
                img = img_data

            if img is None:
                continue

            results = model(img, conf=conf_threshold)
            boxes = results[0].boxes

            detected = []
            if boxes is not None and len(boxes) > 0:
                for box in boxes:
                    cls = int(box.cls[0])
                    conf = float(box.conf[0])
                    x_center = float(box.xywh[0][0])
                    detected.append((x_center, cls, conf))

            if not detected:
                full_number = 'N/A'
                avg_conf = 0.0
            else:
                detected.sort(key=lambda x: x[0])
                digits = [str(d[1]) for d in detected]
                confidences = [d[2] for d in detected]
                full_number = ''.join(digits)
                avg_conf = sum(confidences) / len(confidences)

            frame_num = extract_frame_number(name)
            time_sec = frame_num if frame_num is not None else idx

            results_data.append([time_sec, full_number, f"{avg_conf:.3f}"])

            if save_images and detected:
                annotated_img = results[0].plot()
                is_success, buffer = cv2.imencode(".jpg", annotated_img)
                if is_success:
                    result_images[f"result_{time_sec:04d}_{name}"] = buffer.tobytes()

        status_text.text("✅ 处理完成！")
        progress_bar.empty()
        return results_data, result_images, frame_images

    def process_video(video_bytes, model, fps, save_images, save_frames, conf_threshold):
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
            tmp_file.write(video_bytes)
            tmp_path = tmp_file.name
        
        cap = cv2.VideoCapture(tmp_path)
        if not cap.isOpened():
            raise ValueError("无法打开视频文件")
        
        video_fps = cap.get(cv2.CAP_PROP_FPS)
        if video_fps <= 0:
            video_fps = 25.0
        
        precise_frame_count = 0
        while True:
            ret, _ = cap.read()
            if not ret:
                break
            precise_frame_count += 1
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        total_frames = precise_frame_count
        
        frame_interval = 1 if fps >= video_fps else int(video_fps / fps)
        
        results_data = []
        result_images = {}
        frame_images = {}
        
        frame_count = 0
        extracted_count = 0
        
        progress_bar = st.progress(0, text="正在抽帧并识别...")
        status_text = st.empty()
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_count % frame_interval == 0:
                status_text.text(f"处理中: {frame_count}/{total_frames}")
                progress_bar.progress(frame_count / total_frames if total_frames > 0 else 0)
                
                time_sec = int(frame_count / video_fps)
                
                results = model(frame, conf=conf_threshold)
                boxes = results[0].boxes
                
                detected = []
                if boxes is not None and len(boxes) > 0:
                    for box in boxes:
                        cls = int(box.cls[0])
                        conf = float(box.conf[0])
                        x_center = float(box.xywh[0][0])
                        detected.append((x_center, cls, conf))
                
                if not detected:
                    full_number = 'N/A'
                    avg_conf = 0.0
                else:
                    detected.sort(key=lambda x: x[0])
                    digits = [str(d[1]) for d in detected]
                    confidences = [d[2] for d in detected]
                    full_number = ''.join(digits)
                    avg_conf = sum(confidences) / len(confidences)
                
                results_data.append([time_sec, full_number, f"{avg_conf:.3f}"])
                extracted_count += 1
                
                if save_images and detected:
                    annotated_img = results[0].plot()
                    is_success, buffer = cv2.imencode(".jpg", annotated_img)
                    if is_success:
                        result_images[f"result_{time_sec:04d}.jpg"] = buffer.tobytes()
                        if len(result_images) > 200:
                            oldest_key = list(result_images.keys())[0]
                            del result_images[oldest_key]
                
                if save_frames:
                    is_success, buffer = cv2.imencode(".jpg", frame)
                    if is_success:
                        frame_images[f"original_{time_sec:04d}.jpg"] = buffer.tobytes()
                        if len(frame_images) > 200:
                            oldest_key = list(frame_images.keys())[0]
                            del frame_images[oldest_key]
            
            frame_count += 1
        
        cap.release()
        os.unlink(tmp_path)
        
        st.info(f"📁 从视频中抽取并识别了 {extracted_count} 帧图片")
        
        if len(result_images) >= 200 or len(frame_images) >= 200:
            st.warning("⚠️ 为节省内存，结果图片仅保留最后200张。")
        
        return results_data, result_images, frame_images

    results_data = None
    result_images = {}
    frame_images = {}
    image_files = {}

    if input_type == "📁 图片压缩包 (ZIP)":
        uploaded_file = st.file_uploader(
            "上传图片压缩包 (ZIP)",
            type=['zip'],
            help="请将图片打包成 ZIP 格式上传"
        )
        
        if uploaded_file is not None:
            with st.spinner("📦 正在解压 ZIP 文件..."):
                image_files = {}
                with zipfile.ZipFile(io.BytesIO(uploaded_file.read())) as zip_ref:
                    for file_info in zip_ref.infolist():
                        if file_info.is_dir():
                            continue
                        ext = Path(file_info.filename).suffix.lower()
                        if ext in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']:
                            try:
                                image_files[file_info.filename] = zip_ref.read(file_info.filename)
                            except Exception as e:
                                st.warning(f"无法读取文件: {file_info.filename}, 错误: {e}")
            
            if not image_files:
                st.error("❌ ZIP 包中未找到任何支持的图片文件。")
                st.stop()
            
            st.info(f"📁 共找到 {len(image_files)} 张图片")
            results_data, result_images, frame_images = process_images(
                image_files, model, save_images, conf_threshold
            )

    else:
        uploaded_video = st.file_uploader(
            "上传视频文件",
            type=['mp4', 'avi', 'mov', 'mkv', 'flv', 'wmv'],
            help="支持的格式: MP4, AVI, MOV, MKV, FLV, WMV"
        )
        
        if uploaded_video is not None:
            try:
                results_data, result_images, frame_images = process_video(
                    uploaded_video.read(),
                    model,
                    fps_choice,
                    save_images,
                    save_frames,
                    conf_threshold
                )
            except Exception as e:
                st.error(f"❌ 处理视频时出错: {e}")
                import traceback
                st.code(traceback.format_exc())
                st.stop()

    if results_data:
        if not results_data:
            st.error("❌ 未能识别出任何有效数据。")
            st.stop()
        
        st.subheader("📊 识别结果预览")
        df = pd.DataFrame(results_data, columns=['时间(s)', '电动势', '置信度'])
        st.dataframe(df.head(20), use_container_width=True)
        
        valid_count = len([r for r in results_data if r[1] != 'N/A'])
        st.caption(f"有效识别: {valid_count} / {len(results_data)} 张")
        
        if generate_plot and len(results_data) > 1:
            st.subheader("📈 电动势-时间平滑曲线")
            try:
                valid_data = [row for row in results_data if row[1] != 'N/A']
                if len(valid_data) >= 4:
                    times = [float(row[0]) for row in valid_data]
                    emfs = [float(row[1]) for row in valid_data]
                    confs = [float(row[2]) for row in valid_data]
                    
                    times = np.array(times)
                    emfs = np.array(emfs)
                    confs = np.array(confs)
                    sort_idx = np.argsort(times)
                    times_sorted = times[sort_idx]
                    emfs_sorted = emfs[sort_idx]
                    confs_sorted = confs[sort_idx]
                    
                    filter_mask = (emfs_sorted >= 100) & (emfs_sorted <= 1000)
                    times_plot = times_sorted[filter_mask]
                    emfs_plot = emfs_sorted[filter_mask]
                    confs_plot = confs_sorted[filter_mask]
                    
                    if len(times_plot) >= 4:
                        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
                        scatter = ax1.scatter(times_plot, emfs_plot, c=confs_plot, cmap='viridis', s=20, alpha=0.6)
                        ax1.plot(times_plot, emfs_plot, 'b--', alpha=0.3)
                        ax1.set_xlabel('Time (s)')
                        ax1.set_ylabel('EMF (mV)')
                        ax1.set_title('EMF vs Time - Raw Data')
                        ax1.grid(True, alpha=0.3)
                        plt.colorbar(scatter, ax=ax1, label='Confidence')
                        
                        x_smooth = np.linspace(times_plot.min(), times_plot.max(), 300)
                        spl = make_interp_spline(times_plot, emfs_plot, k=min(3, len(times_plot)-1))
                        y_smooth = spl(x_smooth)
                        ax2.plot(x_smooth, y_smooth, 'r-', linewidth=2, label='Smooth Curve')
                        ax2.scatter(times_plot, emfs_plot, color='blue', s=20, label='Raw Data')
                        ax2.set_xlabel('Time (s)')
                        ax2.set_ylabel('EMF (mV)')
                        ax2.set_title('EMF vs Time - Smooth Curve')
                        ax2.legend()
                        ax2.grid(True, alpha=0.3)
                        plt.tight_layout()
                        st.pyplot(fig)
                    else:
                        st.warning("过滤后有效数据点不足，无法生成平滑曲线。")
                else:
                    st.warning("有效数据点不足（至少需要4个），无法生成平滑曲线。")
            except Exception as e:
                st.warning(f"生成曲线图时出错: {e}")
        
        st.subheader("📥 下载结果")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            csv_buffer = io.StringIO()
            writer = csv.writer(csv_buffer)
            writer.writerow(['时间(s)', '电动势', '置信度'])
            writer.writerows(results_data)
            st.download_button(
                label="📊 下载 CSV 结果",
                data=csv_buffer.getvalue(),
                file_name="recognition_results.csv",
                mime="text/csv",
                use_container_width=True
            )
        
        with col2:
            if result_images:
                zip_buffer = io.BytesIO()
                with zipfile.ZipFile(zip_buffer, 'w') as zip_out:
                    for fname, data in result_images.items():
                        zip_out.writestr(fname, data)
                st.download_button(
                    label="🖼️ 下载结果图片 (ZIP)",
                    data=zip_buffer.getvalue(),
                    file_name="result_images.zip",
                    mime="application/zip",
                    use_container_width=True
                )
            else:
                st.button("🖼️ 下载结果图片 (无)", disabled=True, use_container_width=True)
        
        with col3:
            if frame_images:
                zip_buffer = io.BytesIO()
                with zipfile.ZipFile(zip_buffer, 'w') as zip_out:
                    for fname, data in frame_images.items():
                        zip_out.writestr(fname, data)
                st.download_button(
                    label="🖼️ 下载抽帧原图 (ZIP)",
                    data=zip_buffer.getvalue(),
                    file_name="extracted_frames.zip",
                    mime="application/zip",
                    use_container_width=True
                )
            else:
                st.button("🖼️ 下载抽帧原图 (无)", disabled=True, use_container_width=True)
        
        st.success("🎉 所有任务完成！")


# ============================================================
# 板块2：B-Z振荡反应分析（上传单个CSV文件）
# ============================================================
def page_bz_analysis():
    st.header("⚡ B-Z 振荡反应数据分析 (单文件)")
    st.markdown("上传包含B-Z振荡反应数据的CSV文件（每个文件以温度命名），系统将自动提取诱导期、振荡周期并生成拟合图。")
    
    with st.sidebar:
        st.header("⚙️ B-Z分析参数")
        POTENTIAL_LOW = st.number_input("电势下限 (mV)", value=100, min_value=0, max_value=500)
        POTENTIAL_HIGH = st.number_input("电势上限 (mV)", value=1000, min_value=500, max_value=2000)
        N_VALLEY = st.number_input("谷值数量", value=6, min_value=2, max_value=20)
        N_PEAK = st.number_input("峰值数量", value=6, min_value=2, max_value=20)
        DELTA = st.number_input("波动阈值 (mV)", value=2.0, min_value=0.1, max_value=10.0, step=0.1)
    
    uploaded_files = st.file_uploader(
        "上传CSV文件 (支持多选)",
        type=['csv'],
        accept_multiple_files=True,
        help="每个CSV文件应以温度命名，如 25.csv, 30.5.csv"
    )

    if uploaded_files:
        all_rows = []
        induction_times = {}
        valley_periods = {}
        peak_periods = {}
        time_series_data = {}
        
        progress_bar = st.progress(0, text="处理文件中...")
        status_text = st.empty()
        
        for idx, file in enumerate(uploaded_files):
            status_text.text(f"处理: {file.name} [{idx+1}/{len(uploaded_files)}]")
            progress_bar.progress((idx + 1) / len(uploaded_files))
            
            temp = parse_temperature_from_file(file.name)
            if temp is None:
                st.warning(f"跳过 {file.name}: 无法解析温度")
                continue
            
            try:
                df = pd.read_csv(file)
            except Exception as e:
                st.warning(f"跳过 {file.name}: 无法读取CSV - {e}")
                continue
            
            result = extract_bz_from_dataframe(
                df, temp,
                POTENTIAL_LOW, POTENTIAL_HIGH,
                N_VALLEY, N_PEAK, DELTA
            )
            
            if result[0] is None:
                st.warning(f"跳过 {file.name}: 无有效数据")
                continue
            
            temp_val, t_list, e_list, induction_time, valley_times, peak_times = result
            
            for tt, ee in zip(t_list, e_list):
                all_rows.append({"T": temp_val, "t": tt, "E": ee})
            
            induction_times[temp_val] = induction_time
            time_series_data[temp_val] = (t_list, e_list)
            
            v_period, p_period, _, _ = calculate_periods(valley_times, peak_times)
            valley_periods[temp_val] = v_period
            peak_periods[temp_val] = p_period
        
        status_text.text("✅ 处理完成！")
        progress_bar.empty()
        
        if not all_rows:
            st.error("未提取到有效数据")
            return
        
        # 显示结果
        st.subheader("📊 提取结果")
        df = pd.DataFrame(all_rows)
        st.dataframe(df, use_container_width=True)
        
        # 统计信息
        st.subheader("📈 统计信息")
        stats_data = []
        for temp in sorted(induction_times.keys()):
            stats_data.append({
                'Temperature (°C)': temp,
                'Induction Time (s)': induction_times[temp],
                'Valley Period (s)': valley_periods.get(temp, np.nan),
                'Peak Period (s)': peak_periods.get(temp, np.nan)
            })
        stats_df = pd.DataFrame(stats_data)
        st.dataframe(stats_df, use_container_width=True)
        
        # 绘图
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📉 电势-时间曲线")
            figs = plot_time_series_single(time_series_data)
            for temp, fig in figs.items():
                st.pyplot(fig)
        
        with col2:
            st.subheader("📈 阿伦尼乌斯拟合")
            arrhenius_fig, arrhenius_results = plot_arrhenius(induction_times, valley_periods, peak_periods)
            if arrhenius_fig is not None:
                st.pyplot(arrhenius_fig)
                
                # 显示拟合结果
                st.subheader("📊 拟合结果")
                for key, result in arrhenius_results.items():
                    with st.expander(f"{key} 拟合详情"):
                        st.write(f"方程: {result['equation']}")
                        st.write(f"活化能 Ea = {format_sigfigs(result['Ea_kJ'], 4)} kJ/mol")
                        st.write(f"R² = {format_sigfigs(result['r_squared'], 4)}")
            else:
                st.warning("至少需要2个不同温度的数据才能进行阿伦尼乌斯拟合")
        
        # 下载按钮
        st.subheader("📥 下载结果")
        col1, col2 = st.columns(2)
        
        with col1:
            csv_buffer = io.StringIO()
            df.to_csv(csv_buffer, index=False)
            st.download_button(
                label="📊 下载提取数据 CSV",
                data=csv_buffer.getvalue(),
                file_name="bz_extracted_data.csv",
                mime="text/csv",
                use_container_width=True
            )
        
        with col2:
            stats_csv = io.StringIO()
            stats_df.to_csv(stats_csv, index=False)
            st.download_button(
                label="📊 下载统计信息 CSV",
                data=stats_csv.getvalue(),
                file_name="bz_statistics.csv",
                mime="text/csv",
                use_container_width=True
            )


# ============================================================
# 板块3：整合CSV分析
# ============================================================
def page_integrated_csv_analysis():
    st.header("📊 整合CSV分析")
    st.markdown("""
    上传已整合的CSV文件（如 `bz_extracted_all.csv`），系统将自动按温度分组绘制电势-时间曲线，
    并生成三张阿伦尼乌斯拟合图（诱导期、谷值法、峰值法）。
    
    **CSV格式要求：**
    - 必须包含列：`T` (温度), `t` (时间), `E` (电动势)
    - 每个温度的数据点应包含诱导期和振荡期的完整数据
    """)
    
    with st.sidebar:
        st.header("⚙️ B-Z分析参数")
        delta_value = st.number_input("波动阈值 (mV)", value=2.0, min_value=0.1, max_value=10.0, step=0.1)
        R = 8.314  # 气体常数
    
    uploaded_file = st.file_uploader(
        "上传整合CSV文件",
        type=['csv'],
        help="上传包含 T, t, E 三列的CSV文件"
    )

    if uploaded_file is not None:
        try:
            df_full = pd.read_csv(uploaded_file)
        except Exception as e:
            st.error(f"❌ 无法读取CSV文件: {e}")
            return
        
        # 检查必需的列
        required_cols = ['T', 't', 'E']
        missing_cols = [col for col in required_cols if col not in df_full.columns]
        if missing_cols:
            st.error(f"❌ CSV文件缺少必需的列: {missing_cols}")
            st.info(f"当前列: {list(df_full.columns)}")
            return
        
        st.success(f"✅ 成功读取CSV文件，共 {len(df_full)} 行数据")
        
        # 获取所有温度
        temps = sorted(df_full['T'].unique())
        st.info(f"📊 发现 {len(temps)} 个温度: {[f'{t:.1f}°C' for t in temps]}")
        
        # ===== 1. 绘制每个温度的电势-时间曲线 =====
        st.subheader("📉 电势-时间曲线")
        
        for idx, temp in enumerate(temps):
            df_temp = df_full[df_full['T'] == temp].sort_values('t')
            time_arr = df_temp['t'].values
            pot_arr = df_temp['E'].values
            
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(time_arr, pot_arr, 'b-o', markersize=6, markerfacecolor='b', 
                   markeredgecolor='b', linewidth=1.5, label=f'T = {temp:.1f} °C')
            ax.set_xlabel('Time / s')
            ax.set_ylabel('Potential / mV')
            ax.set_title(f'T = {temp:.1f} °C')
            ax.grid(True, alpha=0.3)
            ax.legend(loc='best')
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)
        
        # ===== 2. 计算每个温度的诱导期和振荡周期 =====
        st.subheader("📈 统计信息")
        
        induction_times = {}
        valley_periods = {}
        peak_periods = {}
        valley_time_details = {}
        peak_time_details = {}
        
        delta = delta_value
        
        for temp in temps:
            df_temp = df_full[df_full['T'] == temp].sort_values('t')
            time_arr = df_temp['t'].values
            pot_arr = df_temp['E'].values
            
            # 找诱导期结束点（第一个谷值）
            min_idx, min_val = find_induction_min(pot_arr, delta)
            induction_time = time_arr[min_idx]
            induction_times[temp] = induction_time
            
            # 从诱导期结束点开始找峰谷（交替提取：谷→峰→谷→峰...）
            # 诱导期结束点本身是第一个谷值
            valley_times = [induction_time]
            peak_times = []
            current_idx = min_idx + 1
            
            n_valley_needed = 6  # 总共需要6个谷值（包含诱导期结束点）
            n_peak_needed = 6    # 总共需要6个峰值
            
            # 交替提取
            while len(peak_times) < n_peak_needed and len(valley_times) < n_valley_needed:
                if current_idx >= len(time_arr):
                    break
                
                # 找下一个峰值
                if len(peak_times) < n_peak_needed:
                    idx, tp, pp = find_next_peak(time_arr, pot_arr, current_idx, delta)
                    if idx < len(time_arr):
                        peak_times.append(tp)
                        current_idx = idx + 1
                    else:
                        break
                
                # 找下一个谷值
                if len(valley_times) < n_valley_needed and current_idx < len(time_arr):
                    idx, tv, pv = find_next_valley(time_arr, pot_arr, current_idx, delta)
                    if idx < len(time_arr):
                        valley_times.append(tv)
                        current_idx = idx + 1
                    else:
                        break
            
            # 存储峰谷时间用于调试
            valley_time_details[temp] = valley_times
            peak_time_details[temp] = peak_times
            
            # 计算振荡周期
            # 谷值法：从第一个谷值到最后一个谷值的总时间 ÷ (谷值数量 - 1)
            if len(valley_times) >= 2:
                total_time = valley_times[-1] - valley_times[0]
                n_intervals = len(valley_times) - 1
                valley_period = total_time / n_intervals
            else:
                valley_period = np.nan
            
            # 峰值法：从第一个峰值到最后一个峰值的总时间 ÷ (峰值数量 - 1)
            if len(peak_times) >= 2:
                total_time = peak_times[-1] - peak_times[0]
                n_intervals = len(peak_times) - 1
                peak_period = total_time / n_intervals
            else:
                peak_period = np.nan
            
            valley_periods[temp] = valley_period
            peak_periods[temp] = peak_period
        
        # 显示统计信息
        stats_data = []
        for temp in sorted(induction_times.keys()):
            stats_data.append({
                'Temperature (°C)': temp,
                'Induction Time (s)': induction_times[temp],
                'Valley Period (s)': valley_periods.get(temp, np.nan),
                'Peak Period (s)': peak_periods.get(temp, np.nan)
            })
        stats_df = pd.DataFrame(stats_data)
        st.dataframe(stats_df, use_container_width=True)
        
        # 显示峰谷时间详情（折叠）
        with st.expander("📋 峰谷时间详情"):
            for temp in sorted(valley_time_details.keys()):
                st.write(f"**T = {temp:.1f} °C**")
                st.write(f"谷值时间: {valley_time_details[temp]}")
                st.write(f"峰值时间: {peak_time_details[temp]}")
                st.write("---")
        
        # ===== 3. 阿伦尼乌斯拟合（三张图） =====
        st.subheader("📈 阿伦尼乌斯拟合")
        
        def temp_to_kelvin(temp_c):
            return temp_c + 273.15
        
        # 三种数据集
        datasets = [
            ('Induction Period', induction_times, '#2E86AB'),
            ('Valley Method', valley_periods, '#A23B72'),
            ('Peak Method', peak_periods, '#F18F01')
        ]
        
        # 创建三列显示三张图
        cols = st.columns(3)
        
        for col_idx, (label, data_dict, color) in enumerate(datasets):
            with cols[col_idx]:
                # 准备数据
                temps_available = [t for t in temps if t in data_dict and not np.isnan(data_dict[t])]
                
                if len(temps_available) >= 2:
                    values = [data_dict[t] for t in temps_available]
                    T_kelvin = [temp_to_kelvin(t) for t in temps_available]
                    x_data = [1.0 / tk for tk in T_kelvin]
                    y_data = np.log(1.0 / np.array(values))
                    
                    # 线性拟合
                    slope, intercept, r_value, p_value, std_err = stats.linregress(x_data, y_data)
                    
                    # 计算活化能
                    Ea = -slope * R
                    Ea_kJ = Ea / 1000
                    
                    # 绘制拟合图
                    fig, ax = plt.subplots(figsize=(5.5, 4.5))
                    
                    x_fit = np.linspace(min(x_data), max(x_data), 100)
                    y_fit = slope * x_fit + intercept
                    
                    ax.scatter(x_data, y_data, color=color, s=60, zorder=5, label='Experimental')
                    ax.plot(x_fit, y_fit, color=color, linestyle='--', linewidth=2, 
                           label=f'ln(1/t) = {slope:.3f}·(1/T) + {intercept:.3f}')
                    
                    ax.set_xlabel('1 / T (K⁻¹)')
                    ax.set_ylabel('ln(1/t)')
                    ax.set_title(f'{label}\nEa = {Ea_kJ:.2f} kJ/mol, R² = {r_value**2:.4f}')
                    ax.grid(True, alpha=0.3)
                    ax.legend(loc='best', fontsize=8)
                    plt.tight_layout()
                    st.pyplot(fig)
                    plt.close(fig)
                else:
                    st.warning(f"数据不足 (需要≥2个温度)")
        
        # ===== 下载按钮 =====
        st.subheader("📥 下载结果")
        
        stats_csv = io.StringIO()
        stats_df.to_csv(stats_csv, index=False)
        st.download_button(
            label="📊 下载统计信息 CSV",
            data=stats_csv.getvalue(),
            file_name="bz_statistics.csv",
            mime="text/csv",
            use_container_width=True
        )
# ============================================================
# 路由
# ============================================================
if page == "📟 数码管数字识别":
    page_digital_tube()
elif page == "⚡ B-Z振荡反应分析":
    page_bz_analysis()
elif page == "📊 整合CSV分析":
    page_integrated_csv_analysis()
