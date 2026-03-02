import os
import sys
import glob

# ============================================================
# ⚙️  CẤU HÌNH - Chỉnh sửa các đường dẫn tại đây
# ============================================================

# THAY ĐỔI: Đường dẫn đến THƯ MỤC chứa nhiều file ảnh
INPUT_DIR = './data' 

# Thư mục gốc của SynthSeg
SYNTHSEG_ROOT = '.'   

# Đường dẫn đến file model và nhãn
PATH_MODEL = 'synthseg_1.0.h5'
PATH_SEG_LABELS  = 'synthseg_segmentation_labels.npy'
PATH_SEG_NAMES   = 'synthseg_segmentation_names.npy'
PATH_TOPO_CLASSES = 'synthseg_topological_classes.npy'

# Thư mục output chung
OUTPUT_DIR = './synthseg_batch_outputs'

# ============================================================
# ⚙️  THAM SỐ MÔ HÌNH (Giữ nguyên)
# ============================================================
CROPPING         = 192
TARGET_RES       = 1.0
FLIP             = True
N_NEUTRAL_LABELS = 18
SIGMA_SMOOTHING  = 0.5
KEEP_BIGGEST     = True
N_LEVELS         = 5
NB_CONV_PER_LEVEL = 2
CONV_SIZE        = 3
UNET_FEAT_COUNT  = 24
FEAT_MULTIPLIER  = 2
ACTIVATION       = 'elu'

# ============================================================
# 🚀  CHẠY BATCH
# ============================================================

def main():
    # 1. Setup môi trường
    synthseg_root = os.path.abspath(SYNTHSEG_ROOT)
    if synthseg_root not in sys.path:
        sys.path.insert(0, synthseg_root)

    # 2. Kiểm tra folder input và các file model
    if not os.path.isdir(INPUT_DIR):
        print(f"❌ Không tìm thấy thư mục đầu vào: {INPUT_DIR}")
        sys.exit(1)

    # Tìm tất cả file ảnh hỗ trợ
    extensions = ['*.nii', '*.nii.gz', '*.mgz']
    image_files = []
    for ext in extensions:
        image_files.extend(glob.glob(os.path.join(INPUT_DIR, ext)))

    if not image_files:
        print(f"❌ Không tìm thấy file .nii, .nii.gz hoặc .mgz nào trong {INPUT_DIR}")
        sys.exit(1)

    print(f"🔍 Tìm thấy {len(image_files)} file cần xử lý.")

    # 3. Tạo thư mục output
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    seg_dir  = os.path.join(OUTPUT_DIR, 'segmentations')
    info_dir = os.path.join(OUTPUT_DIR, 'info')
    os.makedirs(seg_dir,  exist_ok=True)
    os.makedirs(info_dir, exist_ok=True)

    # 4. Import SynthSeg
    try:
        from predict import predict
    except ImportError as e:
        print(f"\n❌ Không thể import SynthSeg: {e}")
        sys.exit(1)

    # 5. Vòng lặp xử lý từng file
    for i, img_path in enumerate(image_files):
        basename = os.path.basename(img_path)
        # Tách tên file và đuôi
        if img_path.endswith('.nii.gz'):
            stem = basename.replace('.nii.gz', '')
            ext = '.nii.gz'
        else:
            stem, ext = os.path.splitext(basename)

        print(f"\n--- [{i+1}/{len(image_files)}] Đang xử lý: {basename} ---")

        path_seg        = os.path.join(seg_dir,  f'{stem}_synthseg{ext}')
        path_posteriors = os.path.join(info_dir, f'{stem}_posteriors.nii.gz')
        path_volumes    = os.path.join(info_dir, f'{stem}_volumes.csv')

        # Kiểm tra nếu đã tồn tại thì bỏ qua (nếu muốn chạy lại từ đầu thì comment phần này)
        if os.path.exists(path_seg):
            print(f"   ⏩ Đã tồn tại kết quả cho {basename}, bỏ qua.")
            continue

        try:
            predict(
                path_images          = img_path,
                path_segmentations   = path_seg,
                path_model           = PATH_MODEL,
                labels_segmentation  = PATH_SEG_LABELS,
                n_neutral_labels     = N_NEUTRAL_LABELS,
                names_segmentation   = PATH_SEG_NAMES if os.path.isfile(PATH_SEG_NAMES) else None,
                path_posteriors      = path_posteriors,
                path_resampled       = None,
                path_volumes         = path_volumes,
                cropping             = CROPPING,
                target_res           = TARGET_RES,
                flip                 = FLIP,
                topology_classes     = PATH_TOPO_CLASSES if os.path.isfile(PATH_TOPO_CLASSES) else None,
                sigma_smoothing      = SIGMA_SMOOTHING,
                keep_biggest_component = KEEP_BIGGEST,
                n_levels             = N_LEVELS,
                nb_conv_per_level    = NB_CONV_PER_LEVEL,
                conv_size            = CONV_SIZE,
                unet_feat_count      = UNET_FEAT_COUNT,
                feat_multiplier      = FEAT_MULTIPLIER,
                activation           = ACTIVATION,
                recompute            = True,
                verbose              = False, #
            )
            print(f"   ✅ Xong: {basename}")
        except Exception as e:
            print(f"   ❌ Lỗi khi xử lý {basename}: {e}")

    print("\n" + "="*30)
    print("🏁 TẤT CẢ ĐÃ HOÀN THÀNH!")
    print(f"Kết quả lưu tại: {os.path.abspath(OUTPUT_DIR)}")
    print("="*30)

if __name__ == '__main__':
    main()