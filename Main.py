import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.vgg19 import preprocess_input

# 加载 Keras 模型
model = load_model("mymodel.keras")

# 类别与索引映射
class_names = {
    0: 'African Violet (Saintpaulia ionantha)',
    1: 'Aloe Vera',
    2: 'Anthurium (Anthurium andraeanum)',
    3: 'Areca Palm (Dypsis lutescens)',
    4: 'Asparagus Fern (Asparagus setaceus)',
    5: 'Begonia (Begonia spp.)',
    6: 'Bird of Paradise (Strelitzia reginae)',
    7: 'Birds Nest Fern (Asplenium nidus)',
    8: 'Boston Fern (Nephrolepis exaltata)',
    9: 'Calathea',
    10: 'Cast Iron Plant (Aspidistra elatior)',
    11: 'Chinese Money Plant (Pilea peperomioides)',
    12: 'Chinese evergreen (Aglaonema)',
    13: 'Christmas Cactus (Schlumbergera bridgesii)',
    14: 'Chrysanthemum',
    15: 'Ctenanthe',
    16: 'Daffodils (Narcissus spp.)',
    17: 'Dracaena',
    18: 'Dumb Cane (Dieffenbachia spp.)',
    19: 'Elephant Ear (Alocasia spp.)',
    20: 'English Ivy (Hedera helix)',
    21: 'Hyacinth (Hyacinthus orientalis)',
    22: 'Iron Cross begonia (Begonia masoniana)',
    23: 'Jade plant (Crassula ovata)',
    24: 'Kalanchoe',
    25: 'Lilium (Hemerocallis)',
    26: 'Lily of the valley (Convallaria majalis)',
    27: 'Money Tree (Pachira aquatica)',
    28: 'Monstera Deliciosa (Monstera deliciosa)',
    29: 'Orchid',
    30: 'Parlor Palm (Chamaedorea elegans)',
    31: 'Peace lily',
    32: 'Poinsettia (Euphorbia pulcherrima)',
    33: 'Polka Dot Plant (Hypoestes phyllostachya)',
    34: 'Ponytail Palm (Beaucarnea recurvata)',
    35: 'Pothos (Ivy arum)',
    36: 'Prayer Plant (Maranta leuconeura)',
    37: 'Rattlesnake Plant (Calathea lancifolia)',
    38: 'Rubber Plant (Ficus elastica)',
    39: 'Sago Palm (Cycas revoluta)',
    40: 'Schefflera',
    41: 'Snake plant (Sanseviera)',
    42: 'Tradescantia',
    43: 'Tulip',
    44: 'Venus Flytrap',
    45: 'Yucca',
    46: 'ZZ Plant (Zamioculcas zamiifolia)'
}

# 图像预处理
def preprocess_image(image_path):
    img = Image.open(image_path).convert('RGB').resize((224, 224))  # 假设模型需要 224x224 输入
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)  # 增加批次维度
    img_array = preprocess_input(img_array)  # VGG19 标准化
    return img_array

# 定义预测函数
def predict(image_path):
    img_array = preprocess_image(image_path)
    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions, axis=1)[0]
    return class_names.get(predicted_class, "Unknown")

# 打开文件并预测
def open_image():
    file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.png;*.jpg;*.jpeg")])
    if not file_path:
        return

    try:
        # 显示图像
        img = Image.open(file_path).resize((300, 300))
        img_tk = ImageTk.PhotoImage(img)
        label_image.config(image=img_tk)
        label_image.image = img_tk

        # 执行预测
        predicted_name = predict(file_path)
        label_result.config(text=f"Predicted Plant: {predicted_name}")
    except Exception as e:
        messagebox.showerror("Error", f"Failed to process image: {e}")

# 创建主窗口
root = tk.Tk()
root.title("Plant Classifier")
root.geometry("500x600")
root.config(bg="#f2f2f2")

# 标题
label_title = tk.Label(root, text="Plant Classifier", font=("Arial", 20, "bold"), bg="#f2f2f2", fg="#333")
label_title.pack(pady=10)

# 图片显示区域
label_image = tk.Label(root, bg="#f2f2f2")
label_image.pack(pady=10)

# 打开图片按钮
button_open = tk.Button(root, text="Open Image", command=open_image, font=("Arial", 14), bg="#4CAF50", fg="white")
button_open.pack(pady=10)

# 预测结果显示区域
label_result = tk.Label(root, text="Predicted Plant: N/A", font=("Arial", 16), bg="#f2f2f2", fg="#333")
label_result.pack(pady=20)

# 启动主循环
root.mainloop()
