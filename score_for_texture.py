from utils import TFRecordDataset, init_model_for_testing
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
import numpy as np

image_shape = (1194, 938)
batch_size = 1
max_pixel_value = 4095.0  # TODO: change to the effective max intensity of your sample
max_dicom_shape = (3584, 2816)  # TODO: change to max image size (x,y) of your sample
dicom_resolution = 0.085  # TODO: change to resolution of your sample
target_resolution = 0.255
toggle_normalization = True
target_pixel_mean = 0.29780406
target_pixel_sd = 0.1523885
source_pixel_mean = 0.269245035  # TODO: calculate this based on sample of 1000 views from you own sample
source_pixel_sd = 0.150967025  # TODO: calculate this based on sample of 1000 views from you own sample

dicom_images = [  # TODO: change this to your own sample of dicom images
    'LMLO.DCM',
    'RMLO.DCM',
    'LCC.DCM',
    'RCC.DCM'
]

texture_model_dirs = ['texture_model_%i.hdf5' % model_i for model_i in range(5)]

ensemble_texture_scores = []
for texture_model in texture_model_dirs:

    model = init_model_for_testing(texture_model, image_shape=image_shape)

    prediction_dataset = TFRecordDataset(
        dicom_images, image_shape, batch_size,
        max_pixel_value, max_dicom_shape, dicom_resolution, target_resolution,
        normalize=toggle_normalization, target_pixel_mean=target_pixel_mean, target_pixel_sd=target_pixel_sd,
        source_pixel_mean=source_pixel_mean, source_pixel_sd=source_pixel_sd
    )

    predictions = model.predict(prediction_dataset.get_input_fn())

    ensemble_texture_scores.append(predictions.ravel())

ensemble_texture_scores = np.array(ensemble_texture_scores)

print('Texture score for exam = %.4f ' % ensemble_texture_scores.mean())


# Display input views
ds_numpy = tfds.as_numpy(prediction_dataset.get_input_fn())
for img in ds_numpy:
    plt.imshow(np.squeeze(img), cmap='gray')
    plt.show()




