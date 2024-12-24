from flask import Flask, request, render_template
from PIL import Image
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.vgg19 import preprocess_input

# 初始化 Flask 应用
app = Flask(__name__)

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
def preprocess_image(image):
    img = image.convert('RGB').resize((224, 224))
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    return img_array

# 定义预测函数
def predict(image):
    img_array = preprocess_image(image)
    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions, axis=1)[0]
    return class_names.get(predicted_class, "Unknown")

# 路由：主页
@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        # 检查是否上传了文件
        if "file" not in request.files:
            return "No file part"
        file = request.files["file"]
        if file.filename == "":
            return "No selected file"

        # 加载图像并进行预测
        try:
            image = Image.open(file)
            predicted_name = predict(image)
            return render_template("result.html", prediction=predicted_name)
        except Exception as e:
            return f"Error: {e}"
    return render_template("index.html")

# 启动 Flask 应用
if __name__ == "__main__":
    app.run(debug=True)
