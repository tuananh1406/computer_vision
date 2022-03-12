from argparse import ArgumentParser

import cv2

from common_utils import MEAN_SUBTRACTIONS, post_process_neural_style_image
from common_utils import thiet_lap_logging


# Tạo logger
logger = thiet_lap_logging(__name__)

# parse the script parameters
parser = ArgumentParser()
parser.add_argument("--image", dest="image_path", required=True)
parser.add_argument("--model", dest="model_path", required=True)
args = parser.parse_args()
image_path = args.image_path
model_path = args.model_path
logger.info('Chạy mô hình: %s', model_path)

# load image
logger.info('Đọc ảnh đầu vào')
image = cv2.imread(image_path)
height, width = image.shape[:2]
logger.info('width: %s, height: %s', width, height)
if width > 1000:
    logger.info('Giảm kích cỡ ảnh')
    image = cv2.resize(
        image,
        None,
        fx=0.4,
        fy=0.4,
        interpolation=cv2.INTER_LINEAR,
    )
    height, width = image.shape[:2]
    logger.info('width: %s, height: %s', width, height)

# load model
logger.info('Tải mô hình')
net = cv2.dnn.readNetFromTorch(model_path)

# run model
logger.info('Chạy mô hình')
blob = cv2.dnn.blobFromImage(
    image,
    1.0,
    (width, height),
    MEAN_SUBTRACTIONS,
    swapRB=False,
    crop=False,
)
net.setInput(blob)
logger.info('Chạy forward')
output = net.forward()

# post process the output result
logger.info('Tiền xử lý ảnh kết quả')
output = post_process_neural_style_image(output)

# display image
logger.info('Hiển thị ảnh kết quả')
cv2.imshow("Output", output)
cv2.waitKey(0)
cv2.destroyAllWindows()
