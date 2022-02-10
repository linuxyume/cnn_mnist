import csv
from PIL import Image
import numpy as np
from tensorflow.keras import datasets, layers, models


input_csv_file = "train.csv"

epoch_num = 10

# クラス

def save_image(np_array, file_name:str):
    second_dimention = np_array.reshape((28,28))
    new_image = Image.fromarray(second_dimention)
    new_image.save(file_name)

def print_data_num(data_list:list, theme:str):
    output_str = "{0} num : {1}, {2} header : {3}".format(theme,len(data_list),theme,data_list[0:10])
    print(output_str)


def parse_to_img_and_label(column:list):
    label = np.array(column[0], dtype=np.uint8) 
    label = label.reshape((1))
    one_dimention_np_array = np.array(column[1:], dtype=np.uint8)
    second_dimention = one_dimention_np_array.reshape((28,28,1))
    return label,second_dimention




if __name__ == "__main__":
    label_list = []
    data_list = []

    with open(input_csv_file,"r") as csv_fd:
        csv_data = csv.reader(csv_fd)
        next(csv_data) # 列のラベルを取り去る。
        for row in csv_data:
            label,img_data_np = parse_to_img_and_label(row)
            label_list.append(label)
            data_list.append(img_data_np)
            # save_image(one_dimention, 'new.png')
        label_list = np.array(label_list)
        data_list = np.array(data_list)
        print(np.shape(label_list))
        print(np.shape(data_list))

        # 正規化
        data_list = data_list / 255.0

        model = models.Sequential()
        model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Conv2D(64, (3, 3), activation='relu'))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Conv2D(64, (3, 3), activation='relu'))
        model.add(layers.Flatten())
        model.add(layers.Dense(64, activation='relu'))
        model.add(layers.Dense(10, activation='softmax'))
        model.summary()


        model.compile(optimizer='adam',
                      loss='sparse_categorical_crossentropy',
                      metrics=['accuracy'])

        model.fit(data_list, label_list, epochs=epoch_num)
        test_loss, test_acc = model.evaluate(data_list,  label_list, verbose=2)
        model.save_weights("model")


