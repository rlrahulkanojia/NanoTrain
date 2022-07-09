from src.network import create_network, create_network_predictor
from utils.misc import Timer
import cv2
import sys


if len(sys.argv) < 4:
    print('Usage: python run_ssd_example.py  <model path> <label path> <image path>')
    sys.exit(0)
model_path = sys.argv[1]
label_path = sys.argv[2]
image_path = sys.argv[3]

class_names = [name.strip() for name in open(label_path).readlines()]

net       = create_network(len(class_names), is_test=True)
net.load(model_path)
predictor = create_network_predictor(net, candidate_size=200)

orig_image           = cv2.imread(image_path)
image                = cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB)
boxes, labels, probs = predictor.predict(image, 10, 0.4)

for i in range(boxes.size(0)):
    box = boxes[i, :]
    cv2.rectangle(orig_image, (box[0], box[1]), (box[2], box[3]), (255, 255, 0), 4)
    label = f"{class_names[labels[i]]}: {probs[i]:.2f}"
    cv2.putText(orig_image, label,
                (box[0] + 20, box[1] + 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,  # font scale
                (255, 0, 255),
                2)  # line type
path = "example_output.jpg"
cv2.imwrite(path, orig_image)
print(f"Found {len(probs)} objects. The output image is {path}")
