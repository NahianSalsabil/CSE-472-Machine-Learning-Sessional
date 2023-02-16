import pickle
import sys
from train_1705091 import *


loaded_model = pickle.load(open("1705091_model2.pickle", "rb"))
print("Pickle Model loaded")

path = sys.argv[1]

#read the test images
test_image_list = []
filelist = os.listdir(path)
# print("filelist: ", filelist)

test_csv = pd.read_csv('test.csv')
y_test = test_csv['digit']
y_test = np.array(y_test)
encoded_y_test = One_Hot_Encoding(y_test)

filelist = test_csv['filename']

for file in filelist:
    img = cv2.imread(os.path.join(path, file))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (32, 32))
    img = (255-img.transpose(2, 0, 1))/255
    test_image_list.append(img)
    
    
X_test = np.array(test_image_list)

loaded_model.test(filelist, X_test, encoded_y_test)
    

