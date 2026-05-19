import torch
import os
import sys
# Import 2 hàm từ API của bạn bạn (từ file model.py trong thư mục myAPI)
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from myAPI.mnist_to_pt import mnist_pt, pt_mnist 

def filter_and_save_ubyte(orig_img_path, orig_lbl_path, out_dir):
    # Tạo thư mục output nếu chưa có để gọn gàng
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
        

    images = mnist_pt(orig_img_path)  
    labels = mnist_pt(orig_lbl_path)  

    train_size = 8750
    test_size = 1250

   
    shuffled_indices = torch.randperm(len(images))
    
    train_indices = shuffled_indices[:train_size]
    test_indices = shuffled_indices[train_size : train_size + test_size]

    mini_train_images = images[train_indices]
    mini_train_labels = labels[train_indices]

    mini_test_images = images[test_indices]
    mini_test_labels = labels[test_indices]

    train_img_out = os.path.join(out_dir, "mini-train-images.ubyte")
    train_lbl_out = os.path.join(out_dir, "mini-train-labels.ubyte")
    test_img_out = os.path.join(out_dir, "mini-test-images.ubyte")
    test_lbl_out = os.path.join(out_dir, "mini-test-labels.ubyte")

    pt_mnist(mini_train_images, train_img_out)
    pt_mnist(mini_train_labels, train_lbl_out)
    pt_mnist(mini_test_images, test_img_out)
    pt_mnist(mini_test_labels, test_lbl_out)

    print("\n✅ HOÀN THÀNH XUẤT SẮC!")
    print(f"-> File Train Ảnh: {train_img_out} (Shape: {mini_train_images.shape})")
    print(f"-> File Train Nhãn: {train_lbl_out} (Shape: {mini_train_labels.shape})")
    print(f"-> File Test Ảnh: {test_img_out} (Shape: {mini_test_images.shape})")
    print(f"-> File Test Nhãn: {test_lbl_out} (Shape: {mini_test_labels.shape})")

if __name__ == "__main__":
    ORIGINAL_IMAGES = "train-images-idx3-ubyte"  
    ORIGINAL_LABELS = "train-labels-idx1-ubyte"  
    
    OUTPUT_DIRECTORY = "mini_mnist_data"

    filter_and_save_ubyte(ORIGINAL_IMAGES, ORIGINAL_LABELS, OUTPUT_DIRECTORY)