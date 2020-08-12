class config:
    input_shape = (384, 512, 3)
    learning_rate = 0.001
    num_epoch = 200
    num_classes = 6
    batch_size = 32
    classes = {'cardboard': 0, 'glass': 1, 'metal': 2, 'paper': 3, 'plastic': 4, 'trash': 5}
    data_path = "./datasets/la1ji1fe1nle4ishu4ju4ji22-momodel/dataset-resized/"
    model_path = "./results/res.h5"
    logs_path = "./results/tb_results/tutorial/"
