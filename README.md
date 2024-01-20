# Создание генеративно-состязательных сетей (GAN) для создания изображений лиц

Проект по созданию генеративно-состязательной сети (GAN) для генерации изображений лиц. Разработан с использованием TensorFlow и Keras. В проекте предоставлены примеры кода, инструкции по использованию и изображения, демонстрирующие качество генерации.

<a target="_blank" href="https://colab.research.google.com/github/LisiyLexa/Lab1-GAN-Faces/blob/main/gan.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>

## Технологии и библиотеки

- TensorFlow 2.0
- Keras

## Архитектура GAN

```python
# Генератор
def build_generator(seed_size, channels):
    model = Sequential()

    model.add(Dense(4 * 4 * 256, activation="relu", input_dim=seed_size))
    model.add(Reshape((4, 4, 256)))

    model.add(UpSampling2D())
    model.add(Conv2D(256, kernel_size=3, padding="same"))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Activation("relu"))

    model.add(UpSampling2D())
    model.add(Conv2D(256, kernel_size=3, padding="same"))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Activation("relu"))

    model.add(UpSampling2D())
    model.add(Conv2D(128, kernel_size=3, padding="same"))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Activation("relu"))

    if GENERATE_RES > 1:
      model.add(UpSampling2D(size=(GENERATE_RES,GENERATE_RES)))
      model.add(Conv2D(128, kernel_size=3, padding="same"))
      model.add(BatchNormalization(momentum=0.8))
      model.add(Activation("relu"))

    model.add(Conv2D(channels, kernel_size=3, padding="same"))
    model.add(Activation("tanh"))

    return model

# Дискриминатор
def build_discriminator(image_shape):
    model = Sequential()

    model.add(Conv2D(32, kernel_size=3, strides=2, input_shape=image_shape, padding="same"))
    model.add(LeakyReLU(alpha=0.2))

    model.add(Dropout(0.25))
    model.add(Conv2D(64, kernel_size=3, strides=2, padding="same"))
    model.add(ZeroPadding2D(padding=((0,1),(0,1))))
    model.add(BatchNormalization(momentum=0.8))
    model.add(LeakyReLU(alpha=0.2))

    model.add(Dropout(0.25))
    model.add(Conv2D(128, kernel_size=3, strides=2, padding="same"))
    model.add(BatchNormalization(momentum=0.8))
    model.add(LeakyReLU(alpha=0.2))

    model.add(Dropout(0.25))
    model.add(Conv2D(256, kernel_size=3, strides=1, padding="same"))
    model.add(BatchNormalization(momentum=0.8))
    model.add(LeakyReLU(alpha=0.2))

    model.add(Dropout(0.25))
    model.add(Conv2D(512, kernel_size=3, strides=1, padding="same"))
    model.add(BatchNormalization(momentum=0.8))
    model.add(LeakyReLU(alpha=0.2))

    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(1, activation='sigmoid'))

    return model
```

## Набор данных

Для обучения использовался набор данных [faces_data_new](https://www.kaggle.com/datasets/gasgallo/faces-data-new). Размер: 7864 изображений.

## Обучение

### Функиця обучения
```python
def train_step(images):
  seed = tf.random.normal([BATCH_SIZE, SEED_SIZE])

  with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
    generated_images = generator(seed, training=True)

    real_output = discriminator(images, training=True)
    fake_output = discriminator(generated_images, training=True)

    gen_loss = generator_loss(fake_output)
    disc_loss = discriminator_loss(real_output, fake_output)


    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))
  return gen_loss, disc_loss
```
## Результаты
### 50 эпох
![train-50](https://github.com/LisiyLexa/Lab1-GAN-Faces/assets/81087786/377f048d-8123-42f3-847c-7de5136b330b)

### 100 эпох
![train-100](https://github.com/LisiyLexa/Lab1-GAN-Faces/assets/81087786/e92c1517-24cd-486d-a586-efd97b8e93a6)

### 300 эпох
![train-300](https://github.com/LisiyLexa/Lab1-GAN-Faces/assets/81087786/1a4842db-a1f2-4aeb-b5db-5711caf631ae)

## Запуск кода
### Запуск готовых моделей
В репозитории содержатся сохраненные модели, обученные на 50, 100 и 300 эпохах.

Для запуска достаточно запустить код ниже
```python
import tensorflow as tf
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt

generator = load_model("путь/к_файлу/модели.h5")
noise = tf.random.normal([1, 100])
generated_image = generator.predict(noise)

plt.imshow(generated_image[0, :, :, :] * 0.5 + 0.5)
```
### Запуск обучения на своих изображениях
**Учтите, что для более быстрого обучения нужно использовать видеокарту**

Если вы хотите обучить модель на своем датасете, то вам нужно:
1. открыть файл **gan.ipynb** в удобной для вас среде.
2. Указать нужные настройки в данном блоке
![image](https://github.com/LisiyLexa/Lab1-GAN-Faces/assets/81087786/54acfcf6-c868-40f4-b0a3-f64cdeac7769)
3. Указть в переменной `faces_path` путь к папке с вашими изображениями
![image](https://github.com/LisiyLexa/Lab1-GAN-Faces/assets/81087786/78f72dbe-15a0-4763-add7-0795878381da)
4. Запустить все блоки по порядку

В конце вы увидите результат обучения. Ваша модель будет сохранена в файл `face_generator.h5`. В папке `output` будут сгенерированные изображения для каждой эпохи обучения.

## Проблемы и улучшения
В целом, изображения получаются довольно неплохими. Думаю, для "самопальной" модели такой результат можно считать успехом.

Для улучшения результата предлагаю поигаться с архитектурой модели, настройками функции потерь/оптимизаторов, использовоать другой датасет или увеличить длительность обучения. 
