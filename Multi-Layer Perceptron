import wandb
sweep_config = {
    'method': 'grid',  # or 'random', 'bayes'
    'metric': {
        'name': 'val_accuracy',
        'goal': 'maximize'
    },
    'parameters': {
        'epochs': {'values': [5, 10]},
        'batch_size': {'values': [8, 16]},
        'img_height': {'values': [16, 32]},
        'img_width': {'values': [16, 32]},
        'learning_rate': {'values': [0.001, 0.0001]},
        'hidden_units': {'values': [64, 128]}
    }
}
# Initialize wandb
# wandb.init(project="flowers-classifier-sweep-sep7th")

# config = wandb.config

# # Use config values
# IMG_HEIGHT = config.img_height
# IMG_WIDTH = config.img_width
# BATCH_SIZE = config.batch_size
# EPOCHS = config.epochs
# LEARNING_RATE = config.learning_rate
# HIDDEN_UNITS = config.hidden_units

def train():
    # Initialize a new wandb run for each sweep configuration
    wandb.init(project="flowers-classifier-sweep-sep7th")

    # Access sweep configuration
    config = wandb.config

    # Use config values
    IMG_HEIGHT = config.img_height
    IMG_WIDTH = config.img_width
    BATCH_SIZE = config.batch_size
    EPOCHS = config.epochs
    LEARNING_RATE = config.learning_rate
    HIDDEN_UNITS = config.hidden_units

    # Define datasets
    train_dataset = (
        tf.data.TextLineDataset("gs://cloud-ml-data/img/flower_photos/train_set.csv")
        .map(lambda x: parse_csvline(x, [config.img_height, config.img_width]), num_parallel_calls=tf.data.AUTOTUNE)
        .batch(config.batch_size)
        .prefetch(tf.data.AUTOTUNE)
    )

    eval_dataset = (
        tf.data.TextLineDataset("gs://cloud-ml-data/img/flower_photos/eval_set.csv")
        .map(lambda x: parse_csvline(x, [config.img_height, config.img_width]), num_parallel_calls=tf.data.AUTOTUNE)
        .batch(config.batch_size)
        .prefetch(tf.data.AUTOTUNE)
    )

    model = keras.Sequential([
        keras.layers.Flatten(input_shape=(config.img_height, config.img_width, IMG_CHANNELS)),
        keras.layers.Dense(config.hidden_units, activation="relu"),
        keras.layers.Dense(len(CLASS_NAMES), activation="softmax")
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=config.learning_rate),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
        metrics=["accuracy"]
    )

    history = model.fit(
        train_dataset,
        validation_data=eval_dataset,
        epochs=config.epochs,
        callbacks=[WandbCallback()]
    )

    # Finish the wandb run
    wandb.finish()
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow import keras
from wandb.keras import WandbCallback

# Constants
IMG_HEIGHT = 16
IMG_WIDTH = 16
IMG_CHANNELS = 3
CLASS_NAMES = ["daisy", "dandelion", "roses", "sunflowers", "tulips"]

def read_and_decode(filename, resize_dims):
    img_bytes = tf.io.read_file(filename)
    img = tf.image.decode_jpeg(img_bytes, channels=IMG_CHANNELS)
    img = tf.image.convert_image_dtype(img, tf.float32)
    img = tf.image.resize(img, resize_dims)
    return img

def parse_csvline(csv_line, resize_dims):
    record_default = ["", ""]
    filename, label_string = tf.io.decode_csv(csv_line, record_default)

    img = read_and_decode(filename, resize_dims)

    # Convert label string to an integer index
    label = tf.where(tf.equal(CLASS_NAMES, label_string))[0, 0]

    return img, label

# # Define datasets
# train_dataset = (
#     tf.data.TextLineDataset("gs://cloud-ml-data/img/flower_photos/train_set.csv")
#     .map(parse_csvline, num_parallel_calls=tf.data.AUTOTUNE)
#     .batch(BATCH_SIZE)
#     .prefetch(tf.data.AUTOTUNE)
# )

# eval_dataset = (
#     tf.data.TextLineDataset("gs://cloud-ml-data/img/flower_photos/eval_set.csv")
#     .map(parse_csvline, num_parallel_calls=tf.data.AUTOTUNE)
#     .batch(BATCH_SIZE)
#     .prefetch(tf.data.AUTOTUNE)
# )


# for image_batch, label_batch in train_dataset.take(1):
#     print("Image batch shape:", image_batch.shape)
#     print("Label batch shape:", label_batch.shape)
#     print("Labels:", label_batch.numpy())

# import matplotlib.pyplot as plt

# for image_batch, label_batch in train_dataset.take(2):
#     # Take the first image from the batch
#     first_image = image_batch[0]
#     first_label = label_batch[0]

#     # Convert tensor to numpy array
#     plt.imshow(first_image.numpy())
#     plt.title(f"Label: {CLASS_NAMES[first_label]}")
#     plt.axis('off')
#     plt.show()
# model = keras.Sequential([
#     keras.layers.Flatten(input_shape=(IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS)),
#     keras.layers.Dense(HIDDEN_UNITS, activation="relu"),
#     keras.layers.Dense(len(CLASS_NAMES), activation="softmax")
# ])

# model.compile(
#     optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
#     loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
#     metrics=["accuracy"]
# )
# history = model.fit(
#     train_dataset,
#     validation_data=eval_dataset,
#     epochs=config.epochs,
#     callbacks=[WandbCallback()]
# )
