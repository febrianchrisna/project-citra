import streamlit as st
from PIL import Image
import numpy as np
import cv2
import matplotlib.pyplot as plt
from io import BytesIO

# Variabel global untuk menyimpan gambar hasil filter
filtered_image_pil = None

# Fungsi untuk menampilkan histogram warna
def show_color_histogram(image):
    image_array = np.array(image)
    color_labels = ['Red', 'Green', 'Blue']
    colors = ['r', 'g', 'b']
    
    plt.figure(figsize=(8, 4))
    plt.title("Histogram Warna")
    plt.xlabel("Pixel Intensity")
    plt.ylabel("Frequency")
    
    for i, color in enumerate(colors):
        hist_values = cv2.calcHist([image_array], [i], None, [256], [0, 256])
        plt.plot(hist_values, color=color, label=color_labels[i])
    plt.legend()
    st.pyplot(plt)

# Fungsi untuk menyimpan gambar
def save_image(image, file_format='JPEG'):
    buffer = BytesIO()
    image.save(buffer, format=file_format)
    st.download_button(
        label="Download Image",
        data=buffer.getvalue(),
        file_name=f'edited_image.{file_format.lower()}',
        mime=f'image/{file_format.lower()}'
    )

st.title("Aplikasi Manipulasi Citra dengan Streamlit")

# Pilih gambar
uploaded_file = st.file_uploader("Upload gambar", type=['jpg', 'jpeg', 'png', 'bmp'])
if uploaded_file:
    image = Image.open(uploaded_file).convert('RGB')  # Pastikan format RGB
    st.image(image, caption='Gambar Asli', use_column_width=True)
    st.write("Histogram Asli:")
    show_color_histogram(image)

    # Konversi gambar menjadi format array untuk OpenCV
    img_cv2 = np.array(image)

    # Pilihan fitur manipulasi citra
    st.sidebar.header("Pilih Fitur Manipulasi")
    option = st.sidebar.selectbox("Pilih fitur:", 
                                  ['Grayscale', 'Binary', 'Noise', 'Negative', 'Blur', 'Pseudocolor',
                                   'True Color Filter', 'Edge Detection (Prewitt, Sobel, Canny)'])

    filtered_image_pil = image  # Default ke gambar asli

    if option == 'Grayscale':
        if st.sidebar.button("Apply Grayscale"):
            gray_image = cv2.cvtColor(img_cv2, cv2.COLOR_RGB2GRAY)
            filtered_image_pil = Image.fromarray(gray_image)
            st.image(filtered_image_pil, caption='Gambar Setelah Grayscale', use_column_width=True)
            st.write("Histogram Grayscale:")
            plt.figure(figsize=(8, 4))
            plt.hist(gray_image.ravel(), bins=256, range=(0, 256), color='gray')
            plt.title("Histogram Grayscale")
            plt.xlabel("Pixel Intensity")
            plt.ylabel("Frequency")
            st.pyplot(plt)

    elif option == 'Binary':
        threshold = st.sidebar.slider("Threshold untuk biner", 0, 255, 128)
        if st.sidebar.button("Apply Binary"):
            gray_image = cv2.cvtColor(img_cv2, cv2.COLOR_RGB2GRAY)
            _, binary_image = cv2.threshold(gray_image, threshold, 255, cv2.THRESH_BINARY)
            filtered_image_pil = Image.fromarray(binary_image)
            st.image(filtered_image_pil, caption='Gambar Setelah Biner', use_column_width=True)
            st.write("Histogram Binary:")
            plt.figure(figsize=(8, 4))
            plt.hist(binary_image.ravel(), bins=256, range=(0, 256), color='black')
            plt.title("Histogram Binary")
            plt.xlabel("Pixel Intensity")
            plt.ylabel("Frequency")
            st.pyplot(plt)

    elif option == 'Noise':
        noise_level = st.sidebar.slider("Tingkat Noise (0-100)", 0, 100, 10)
        if st.sidebar.button("Apply Noise"):
            noise = np.random.randint(0, noise_level, img_cv2.shape, dtype='uint8')
            noisy_image = cv2.add(img_cv2, noise)
            # Tidak perlu konversi tambahan
            filtered_image_pil = Image.fromarray(noisy_image)
            st.image(filtered_image_pil, caption='Gambar dengan Noise', use_column_width=True)
            st.write("Histogram Noise:")
            show_color_histogram(filtered_image_pil)

    elif option == 'Negative':
        if st.sidebar.button("Apply Negative"):
            negative_image = cv2.bitwise_not(img_cv2)
            filtered_image_pil = Image.fromarray(negative_image)
            st.image(filtered_image_pil, caption='Gambar Negatif', use_column_width=True)
            st.write("Histogram Negatif:")
            show_color_histogram(filtered_image_pil)

    elif option == 'Blur':
        blur_radius = st.sidebar.slider("Radius Blur", 1, 30, 5)
        if st.sidebar.button("Apply Blur"):
            kernel_size = blur_radius if blur_radius % 2 == 1 else blur_radius + 1
            blurred_image = cv2.GaussianBlur(img_cv2, (kernel_size, kernel_size), 0)
            filtered_image_pil = Image.fromarray(blurred_image)
            st.image(filtered_image_pil, caption='Gambar Setelah Blur', use_column_width=True)
            st.write("Histogram Blur:")
            show_color_histogram(filtered_image_pil)

    elif option == 'Pseudocolor':
        colormap = st.sidebar.selectbox("Pilih Colormap", ['Jet', 'Hot', 'Cool', 'Spring'])
        if st.sidebar.button("Apply Pseudocolor"):
            gray_image = cv2.cvtColor(img_cv2, cv2.COLOR_RGB2GRAY)
            pseudo_image = cv2.applyColorMap(gray_image, getattr(cv2, f'COLORMAP_{colormap.upper()}'))
            filtered_image_pil = Image.fromarray(cv2.cvtColor(pseudo_image, cv2.COLOR_BGR2RGB))
            st.image(filtered_image_pil, caption='Gambar Setelah Pseudocolor', use_column_width=True)
            st.write("Histogram Pseudocolor:")
            show_color_histogram(filtered_image_pil)

    elif option == 'True Color Filter':
        color_filter = st.sidebar.selectbox("Filter Warna", ['Red', 'Green', 'Blue'])
        if st.sidebar.button("Apply True Color Filter"):
            filtered_image = img_cv2.copy()
            if color_filter == 'Red':  # Merah ada di kanal 2 pada format BGR
                filtered_image[:, :, [0, 1]] = 0  # Nolkan Blue dan Green
            elif color_filter == 'Green':  # Hijau ada di kanal 1 pada format BGR
                filtered_image[:, :, [0, 2]] = 0  # Nolkan Blue dan Red
            elif color_filter == 'Blue':  # Biru ada di kanal 0 pada format BGR
                filtered_image[:, :, [1, 2]] = 0  # Nolkan Green dan Red
            filtered_image_pil = Image.fromarray(cv2.cvtColor(filtered_image, cv2.COLOR_BGR2RGB))
            
            # Tampilkan gambar hasil filter
            st.image(filtered_image_pil, caption=f'Gambar Setelah Filter {color_filter}', use_column_width=True)
            
            # Histogram khusus kanal yang difilter
            st.write(f"Histogram Filter {color_filter}:")
            plt.figure(figsize=(8, 4))
            if color_filter == 'Red':
                plt.plot(cv2.calcHist([filtered_image], [2], None, [256], [0, 256]), color='r', label='Red')  # Kanal 2
            elif color_filter == 'Green':
                plt.plot(cv2.calcHist([filtered_image], [1], None, [256], [0, 256]), color='g', label='Green')  # Kanal 1
            elif color_filter == 'Blue':
                plt.plot(cv2.calcHist([filtered_image], [0], None, [256], [0, 256]), color='b', label='Blue')  # Kanal 0
            plt.title(f"Histogram {color_filter}")
            plt.xlabel("Pixel Intensity")
            plt.ylabel("Frequency")
            plt.legend()
            st.pyplot(plt)



    elif option == 'Edge Detection (Prewitt, Sobel, Canny)':
        method = st.sidebar.selectbox("Metode Deteksi Tepi", ['Prewitt', 'Sobel', 'Canny'])
        if method == 'Prewitt':
            if st.sidebar.button("Apply Prewitt"):
                gray_image = cv2.cvtColor(img_cv2, cv2.COLOR_RGB2GRAY)
                prewitt_x = cv2.filter2D(gray_image, -1, np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]]))
                prewitt_y = cv2.filter2D(gray_image, -1, np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]]))
                prewitt_image = cv2.add(prewitt_x, prewitt_y)
                filtered_image_pil = Image.fromarray(prewitt_image)
                st.image(filtered_image_pil, caption='Gambar Setelah Prewitt', use_column_width=True)
                st.write("Histogram Prewitt:")
                plt.figure(figsize=(8, 4))
                plt.hist(prewitt_image.ravel(), bins=256, range=(0, 256), color='black')
                plt.title("Histogram Prewitt")
                plt.xlabel("Pixel Intensity")
                plt.ylabel("Frequency")
                st.pyplot(plt)

        elif method == 'Sobel':
            if st.sidebar.button("Apply Sobel"):
                gray_image = cv2.cvtColor(img_cv2, cv2.COLOR_RGB2GRAY)
                sobel_x = cv2.Sobel(gray_image, cv2.CV_64F, 1, 0, ksize=3)
                sobel_y = cv2.Sobel(gray_image, cv2.CV_64F, 0, 1, ksize=3)
                sobel_image = cv2.magnitude(sobel_x, sobel_y)
                sobel_image = np.uint8(np.absolute(sobel_image))
                filtered_image_pil = Image.fromarray(sobel_image)
                st.image(filtered_image_pil, caption='Gambar Setelah Sobel', use_column_width=True)
                st.write("Histogram Sobel:")
                plt.figure(figsize=(8, 4))
                plt.hist(sobel_image.ravel(), bins=256, range=(0, 256), color='black')
                plt.title("Histogram Sobel")
                plt.xlabel("Pixel Intensity")
                plt.ylabel("Frequency")
                st.pyplot(plt)

        elif method == 'Canny':
            threshold1 = st.sidebar.slider("Threshold 1 untuk Canny", 0, 255, 100)
            threshold2 = st.sidebar.slider("Threshold 2 untuk Canny", 0, 255, 200)
            if st.sidebar.button("Apply Canny"):
                canny_image = cv2.Canny(img_cv2, threshold1, threshold2)
                filtered_image_pil = Image.fromarray(canny_image)
                st.image(filtered_image_pil, caption='Gambar Setelah Canny', use_column_width=True)
                st.write("Histogram Canny:")
                plt.figure(figsize=(8, 4))
                plt.hist(canny_image.ravel(), bins=256, range=(0, 256), color='black')
                plt.title("Histogram Canny")
                plt.xlabel("Pixel Intensity")
                plt.ylabel("Frequency")
                st.pyplot(plt)

    # Simpan hasil
    st.write("Simpan hasil:")
    save_format = st.selectbox("Pilih Format Penyimpanan", ['JPEG', 'PNG'])
    save_image(filtered_image_pil, save_format)
