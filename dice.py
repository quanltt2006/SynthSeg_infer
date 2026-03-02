import nibabel as nib
import numpy as np
import os

def fast_dice(x, y, labels):
    """Tính toán chỉ số Dice nhanh cho danh sách các nhãn"""
    if len(labels) > 1:
        labels_sorted = np.sort(labels)
        label_edges = np.sort(np.concatenate([labels_sorted - 0.1, labels_sorted + 0.1]))
        label_edges = np.insert(label_edges, [0, len(label_edges)], [labels_sorted[0] - 0.1, labels_sorted[-1] + 0.1])

        hst = np.histogram2d(x.flatten(), y.flatten(), bins=label_edges)[0]
        idx = np.arange(start=1, stop=2 * len(labels_sorted), step=2)
        dice_score = 2 * np.diag(hst)[idx] / (np.sum(hst, 0)[idx] + np.sum(hst, 1)[idx] + 1e-5)
        return dice_score
    else:
        return np.array([2 * np.sum((x == labels[0]) * (y == labels[0])) / (np.sum(x == labels[0]) + np.sum(y == labels[0]) + 1e-5)])

if __name__ == "__main__":
    # --- CÀI ĐẶT ĐƯỜNG DẪN TRỰC TIẾP TẠI ĐÂY ---
    # Tôi để mặc định là so sánh kết quả SynthSeg v1 với file aseg truyền thống
    # Nếu bạn chưa có aseg.mgz, hãy đổi tên 'aseg.mgz' thành file bạn muốn so sánh
    path_gt = "frs/synthseg_v1.mgz" 
    path_seg = "synthseg_outputs/segmentations/orig_synthseg.mgz"

    print(f"Đang kiểm tra file...")
    if not os.path.exists(path_gt) or not os.path.exists(path_seg):
        print(f"LỖI: Không tìm thấy file!")
        print(f"Hãy đảm bảo tồn tại:\n1. {path_gt}\n2. {path_seg}")
    else:
        # Load ảnh
        print(f"Đang đọc dữ liệu (có thể mất vài giây)...")
        img1_obj = nib.load(path_gt)
        img2_obj = nib.load(path_seg)
        
        img1 = img1_obj.get_fdata()
        img2 = img2_obj.get_fdata()

        if img1.shape != img2.shape:
            print(f"Lỗi: Kích thước không khớp! {img1.shape} vs {img2.shape}")
        else:
            # Lấy danh sách các nhãn (bỏ qua nhãn 0)
            labels = np.unique(img1)
            labels = labels[labels != 0]

            # Tính Dice
            scores = fast_dice(img1, img2, labels)

            print("\n" + "="*40)
            print(f"KẾT QUẢ SO SÁNH DICE")
            print(f"GT:  {path_gt}")
            print(f"SEG: {path_seg}")
            print("="*40)
            
            # Hiển thị kết quả từng nhãn (rút gọn nếu quá nhiều)
            for i, label in enumerate(labels):
                print(f"Nhãn {int(label):3d}: {scores[i]:.4f}")

            print("-" * 40)
            print(f"DICE TRUNG BÌNH TỔNG: {np.mean(scores):.4f}")
            print("="*40)