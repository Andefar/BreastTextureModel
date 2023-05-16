from os.path import expanduser, exists
import numpy as np
import pydicom
from skimage import filters
import cv2
import SimpleITK as sitk
from classification_models.keras import Classifiers
import tensorflow as tf


class TFRecordDataset:

    def __init__(
            self, input_files, output_shape, batch_size,
            max_pixel_value, max_dicom_shape, dicom_resolution, target_resolution,
            normalize=False,
            target_pixel_mean=0.0, target_pixel_sd=0.0,
            source_pixel_mean=0.0, source_pixel_sd=0.0):

        # Get the filenames belonging to training set
        self.input_filenames = np.array([expanduser(f.strip()) for f in input_files])

        assert all([exists(f) for f in self.input_filenames])

        self.batch_size = batch_size
        self.max_dicom_shape = max_dicom_shape
        self.output_shape = output_shape
        self.max_pixel_value = max_pixel_value

        self.dicom_resolution = dicom_resolution
        self.target_resolution = target_resolution

        self.normalize = normalize
        self.target_pixel_mean = target_pixel_mean
        self.target_pixel_sd = target_pixel_sd
        self.source_pixel_mean = source_pixel_mean
        self.source_pixel_sd = source_pixel_sd

    def __len__(self):
        return int(np.ceil(len(self.input_filenames) / self.batch_size))

    def _parse_dicom(self, image_path):

        image_path = image_path.numpy().decode('utf-8')

        if not exists(expanduser(image_path)):
            raise ValueError('Could find file  %s' % image_path)

        with pydicom.dcmread(expanduser(image_path), force=True) as dicom:

            laterality = dicom.ImageLaterality

            try:
                sitk_reader = sitk.ImageFileReader()
                sitk_reader.SetFileName(image_path)
                image_object = sitk_reader.Execute()
                image = np.squeeze(sitk.GetArrayFromImage(image_object))
            except:
                raise ValueError('Could not load image %s data size=(%i,%i), bits_all=%i, bits_stored=%i' % (
                    image_path, dicom.Rows, dicom.Columns, dicom.BitsAllocated, dicom.BitsStored
                ))

        # Flip image and mask
        if laterality == 'R':
            image = np.flip(image, axis=1)

        if np.all(image == image[0]):
            raise ValueError('Image %s was blank' % image_path)

        # 1st stage padding
        image = np.pad(
            array=image,
            pad_width=((0, self.max_dicom_shape[0] - image.shape[0]), (0, self.max_dicom_shape[1] - image.shape[1])),
            mode='constant',
            constant_values=0
        )

        # Compute size with target resolution
        new_size_x = int(self.max_dicom_shape[0] * (self.dicom_resolution / self.target_resolution))  # Floors the size
        new_size_y = int(self.max_dicom_shape[1] * (self.dicom_resolution / self.target_resolution))  # Floors the size

        # Downscale
        image = cv2.resize(image, dsize=(new_size_y, new_size_x), interpolation=cv2.INTER_LANCZOS4)

        true_output = self.output_shape

        if image.shape != true_output:
            exit('\nSomething went wrong with the resampling/padding.')

        image = image.astype(np.float32)

        # Truncate the intensities.
        image[image > self.max_pixel_value] = self.max_pixel_value
        image[image < 0] = 0

        # Normalize by effective max pixel intensity
        image = image / self.max_pixel_value

        # Make a channel dimension
        image = np.expand_dims(image, axis=-1)

        return image

    def _set_shapes(self, img):
        img.set_shape([self.output_shape[0], self.output_shape[1], 1])
        return img

    def _normalize(self, img):
        if self.normalize:
            background_mask = tf.where(img == 0, 0.0, 1.0)
            img = img - self.source_pixel_mean
            img = tf.multiply(img, (self.target_pixel_sd / self.source_pixel_sd))
            img = img + self.target_pixel_mean
            img = tf.multiply(img, background_mask)
        return img

    def get_input_fn(self):

        dataset = tf.data.Dataset.from_tensor_slices(self.input_filenames)
        dataset = dataset.map(lambda x: tf.py_function(func=self._parse_dicom, inp=[x], Tout=tf.float32))
        dataset = dataset.map(self._set_shapes)
        dataset = dataset.map(self._normalize)
        dataset = dataset.batch(self.batch_size)

        return dataset


class MyChannelRepeat(tf.keras.layers.Layer):
    def __init__(self, repeats, **kwargs):
        self.repeats = int(repeats)
        self.input_spec = [tf.keras.layers.InputSpec(ndim=4)]
        super(MyChannelRepeat, self).__init__(**kwargs)

    def compute_output_shape(self, shape):
        return (shape[0], shape[1], shape[2], shape[3] * self.repeats)

    def call(self, x, mask=None):
        return tf.concat([x, x, x], axis=-1)

    def get_config(self):
        base_config = super(MyChannelRepeat, self).get_config()
        return base_config


def init_model_for_testing(model_dir, image_shape):

    if not exists(expanduser(model_dir)):
        exit('Model %s was not found.' % expanduser(model_dir))

    ResNet, _ = Classifiers.get('seresnet18')
    image_input = tf.keras.Input(shape=image_shape + (1,), dtype=tf.float32, name='image_input')
    image_repeat = MyChannelRepeat(3)(image_input)
    base_model = ResNet(include_top=False, input_shape=image_shape + (3,), weights='imagenet')(image_repeat)
    avg_pool = tf.keras.layers.GlobalAveragePooling2D()(base_model)
    dropout = tf.keras.layers.Dropout(0.5)(avg_pool)
    batch_norm1 = tf.keras.layers.BatchNormalization()(dropout)
    texture_posterior1 = tf.keras.layers.Dense(512, activation='relu', kernel_initializer='he_normal')(batch_norm1)
    dropout = tf.keras.layers.Dropout(0.25)(texture_posterior1)
    batch_norm2 = tf.keras.layers.BatchNormalization()(dropout)
    texture_posterior2 = tf.keras.layers.Dense(1, activation='sigmoid')(batch_norm2)
    model = tf.keras.Model(inputs=image_input, outputs=texture_posterior2)

    model.load_weights(expanduser(model_dir))

    return model
