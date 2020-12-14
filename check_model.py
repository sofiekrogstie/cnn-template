from deoxys.model import model_from_full_config, load_model

import matplotlib.pyplot as plt

if __name__ == '__main__':
    # model = model_from_full_config('config/2d_unet_CT_W_PET.json')

    model = load_model('../../hn_perf/2d_unet/model/model.014.h5')

    # # data handling here

    # model.predict(data)

    model.model.summary()

    res = model.activation_maximization('conv2d_1')


    plt.imshow(res[0][...,0], )

    plt.show()
