from tensorflow.keras import models, layers
from tensorflow import keras
import tensorflow as tf
import streamlit as st


model =keras.models.load_model('Model')
def recommendation(sample):
    sample={name: tf.convert_to_tensor([value]) for name, value in list(sample.items())}
    predictions = model.predict(sample)
    output = tf.nn.sigmoid(predictions[0])
    prob=output*100
    rounded=[float(np.round(x)) for x in prob]
    return rounded
def main():

    st.title('Astra Size Fit recommendation')

    item_id = st.number_input('item_id')
    size = st.number_input('size')
    quality = st.number_input('quality')
    cup_size = st.text_input('cup_size')
    hips = st.text_input('hips')
    bra_size = st.number_input('bra_size')
    category = st.text_input('category')
    height = st.number_input('height')
    user_name = st.text_input('user_name')
    length = st.text_input('length')

    sample = {
    'item_id': item_id,
    'size': size,
    'quality': quality,
    'cup_size': cup_size,
    'hips': hips,
    'bra_size': bra_size,
    'category': category,
    'height': height,
    'user_name': user_name,
    'length': length
    }

    recommend=''

    if st.button('Fit Recommend'):
        recommend=recommendation(sample)

    st.success(recommend)


if __name__=="__main__":
    main()


