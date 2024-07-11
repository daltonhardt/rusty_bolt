import streamlit as st
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
from tempfile import NamedTemporaryFile

# Início do App
st.set_page_config(layout='wide')

# (1) Conteúdo lateral (sidebar)
st.sidebar.title('Área oxidada em parafuso')
arquivo = st.sidebar.file_uploader(label='Selecione o arquivo:', type=['png', 'jpg'])

# (2) Conteúdo principal
# Título
st.title("Identificação de área oxidada")

if arquivo is not None:
    print('Processando...')
    with NamedTemporaryFile(dir='.', suffix='.jpg') as f:  # (,delete=False) para manter o arq temporario
        f.write(arquivo.getbuffer())
        # file_name = os.path.splitext(f.name)[0]
        file_name = f.name
        # print('file_name:', file_name)
        original = cv.imread(file_name)
    st.sidebar.image(arquivo, caption='Arquivo Original')
    rows, cols, channels = original.shape
    original_tot_pixels = original.size
    str_shape = f'rows,cols,channels,total: {rows},{cols},{channels},{original_tot_pixels}'
    st.sidebar.text(str_shape)
    print(f'Original rows,cols,channels,total: {rows},{cols},{channels}, {original_tot_pixels}')

    # Opções avançadas de escolha
    st.sidebar.title('Ampliar imagem:')
    tam_pixels = st.sidebar.slider('Tamanho da imagem [pixels]:', min_value=150, max_value=500, value=210)
    # redimensionando a imagem para X pixels e mantendo o aspect ratio
    file_bytes = np.asarray(bytearray(arquivo.read()), dtype=np.uint8)
    image_decoded = cv.imdecode(file_bytes, cv.IMREAD_ANYCOLOR)
    razao = tam_pixels / image_decoded.shape[1]
    dim = (tam_pixels, int(image_decoded.shape[0] * razao))
    arquivo = cv.resize(image_decoded, dim, interpolation=cv.INTER_CUBIC)
    original = cv.resize(image_decoded, dim, interpolation=cv.INTER_CUBIC)

    # convert to RGB
    img_rgb = cv.cvtColor(original, cv.COLOR_BGR2RGB)
    pixel_colors = img_rgb.reshape((np.shape(img_rgb)[0]*np.shape(img_rgb)[1], 3))
    # pixel values
    norm = colors.Normalize(vmin=-1., vmax=1.)  # normalize all pixels
    norm.autoscale(pixel_colors)
    pixel_colors = norm(pixel_colors).tolist()

    # convert to HSV flatten
    imagem = cv.cvtColor(original, cv.COLOR_BGR2HSV)

    # Opções para HSV
    st.sidebar.title('HSV custom:')
    th_hue = st.sidebar.slider('Hue Threshold:', min_value=0, max_value=255, value=(0, 45))
    th_sat = st.sidebar.slider('Saturation Threshold:', min_value=0, max_value=255, value=(45, 255))
    th_val = st.sidebar.slider('Value Threshold:', min_value=0, max_value=255, value=(0, 255))
    th_hue_min, th_hue_max = th_hue
    th_sat_min, th_sat_max = th_sat
    th_val_min, th_val_max = th_val

    boundaries = [[th_hue_min, th_sat_min, th_val_min], [th_hue_max, th_sat_max, th_val_max]]
    # print('boundaries:', boundaries)
    # create NumPy Arrays from the boundaries
    lower = np.array(boundaries[0], dtype='uint8')
    upper = np.array(boundaries[1], dtype='uint8')
    mask = cv.inRange(imagem, lower, upper)

    # make the mask region to have the same color as the original image we can use bitwise operation
    masked = cv.bitwise_and(imagem, imagem, mask=mask)
    masked_bgr = cv.cvtColor(masked, cv.COLOR_HSV2RGB_FULL)

    # show the images
    col1, col2, col3 = st.columns(3)
    with col1:
        st.image(arquivo, caption='Original BGR-RGB')
    # with col2:
    #     st.image(imagem, caption='Imagem HSV')
    with col2:
        st.image(mask, caption='Máscara')
    with col3:
        st.image(masked_bgr, caption='Area Oxidada')

    r, c, ch = masked_bgr.shape
    summed = np.sum(masked_bgr, axis=2)
    tot_size = masked_bgr.size
    print('Arquivo Masked width, height, channels, total:', c, r, ch, tot_size)
    mask_pixels = cv.countNonZero(mask)
    print('mask_pixels=', mask_pixels)
    area_oxidada = round(mask_pixels * 100 / original_tot_pixels, 2)
    print(f'Area oxidada = {area_oxidada}%')

    # plot 3d projetion
    h, s, v = cv.split(imagem)
    hf, sf, vf = h.flatten(), s.flatten(), v.flatten()  # convert rows and columns to a single row, like a list
    fig = plt.figure()
    axis = fig.add_subplot(111)
    axis.scatter(hf, sf, facecolors=pixel_colors, marker='.')
    axis.set_xlabel('Hue')
    axis.set_ylabel('Saturation')
    # axis.set_zlabel('Value')
    st.pyplot(fig)
