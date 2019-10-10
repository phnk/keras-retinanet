import os

WIDTH = 1920
HEIGHT = 1080

CLASS_MAPPING = {
    "0": "wheel", 
    "1": "cab",
    "2": "tipping body"
}


def read_file(width, height, file_path):
    w, h = width, height
    with open(file_path, 'r') as file:
        lines = file.readlines()
        voc_labels = []
        for line in lines:
            voc = []
            line = line.strip()
            data = line.split()
            bbox_width = float(data[3]) * w
            bbox_height = float(data[4]) * h
            center_x = float(data[1]) * w
            center_y = float(data[2]) * h
            voc.append(round(center_x - (bbox_width / 2)))
            voc.append(round(center_y - (bbox_height / 2)))
            voc.append(round(center_x + (bbox_width / 2)))
            voc.append(round(center_y + (bbox_height / 2)))
            voc.append(CLASS_MAPPING.get(data[0]))
            voc_labels.append(voc)
        print("Processing complete for file: {}".format(file_path))
        return voc_labels

def write_to_file(input_root_dir, dirs, dir, f):
    for files in os.listdir(input_root_dir + dirs + os.sep + dir):
        file_path = input_root_dir + dirs + os.sep + dir + os.sep + files
        if files.split(".")[1] == "txt":
            labels = read_file(WIDTH, HEIGHT, file_path)
            for label in labels:
                label_str = file_path.replace("labels", "images").replace("txt", "png") + "," + ",".join(map(str, label)) + "\n"
                f.write(label_str)
                print("Writing {} to file".format(label_str))


if __name__ == "__main__":
    INPUT_ROOT_DIR = "data/full dataset/"
    OUTPUT_TRAIN_FILE = "train.csv"
    OUTPUT_VALIDATION_FILE = "valid.csv"
 
    for dirs in os.listdir(INPUT_ROOT_DIR):
        for dir in os.listdir(INPUT_ROOT_DIR + dirs):
            if dirs == "train":
                with open(OUTPUT_TRAIN_FILE, "w") as f:
                    write_to_file(INPUT_ROOT_DIR, dirs, dir, f)
            else:
                with open(OUTPUT_VALIDATION_FILE, "w") as f:
                    write_to_file(INPUT_ROOT_DIR, dirs, dir, f)
