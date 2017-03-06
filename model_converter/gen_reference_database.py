import numpy as np
import tensorflow as tf
from sklearn.externals import joblib
import json

from pill_feature_generator import FeatureGenerator

def main():
    output_graph_path = 'model/frozen_model.pb'

    fg = FeatureGenerator(output_graph_path,input_size=227)

    pills = joblib.load('data/pill_imgs_227_color_fb.pkl')

    pill_imgs = []
    pill_names = []
    for pill_img, file_name in zip(pills[0],pills[1]):

        #skip consumer images
        if file_name.find('S')<0:
            continue

        pill_imgs.append(pill_img)
        pill_names.append(file_name)

    pill_imgs = np.array(pill_imgs)
    color_fea,gray_fea = fg.gen_feature(pill_imgs)
    joblib.dump((pill_names,color_fea,gray_fea),'data/ref_db.pkl',compress=3)

    print(color_fea.shape,gray_fea.shape)

def convert_to_json():
    pill_names, color_fea, gray_fea = joblib.load('data/ref_db.pkl')

    json_dict = dict()
    json_dict['ref_pills']=[]
    for name, color, gray in zip(pill_names,color_fea,gray_fea):
        json_dict['ref_pills'].append( [ name ,color.tolist(), gray.tolist()] )

    with open('data/ref_db.json', 'w') as f:
        json.dump(json_dict, f)


if __name__ == '__main__':

    #main()

    convert_to_json()
