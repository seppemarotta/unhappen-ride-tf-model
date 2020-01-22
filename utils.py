from configparser import RawConfigParser
from unidecode import unidecode

class CustomRawConfigParser(RawConfigParser):
    def __repr__(self):
        class_name = type(self).__name__
        r = {k: dict(v) for k, v in dict(self).items()}
        return "{}({})".format(class_name, r)

def read_raw(input_file_name):
    parser = CustomRawConfigParser(converters={'list': parse_list, })
    parser.read(input_file_name)
    return parser

def parse_list(s):
    s = unidecode(s.strip().lower())
    if s:
        return list(map(lambda string: string.strip(), s.split(',')))
    else:
        return []

def plot_history(history):
    import matplotlib.pyplot as plt
    acc = history.history['acc']
    val_acc = history.history['val_acc']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs = range(len(acc))

    plt.plot(epochs, acc, 'bo', label='Training accuracy')
    plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
    plt.title('Training and validation accuracy')

    plt.figure()

    plt.plot(epochs, loss, 'bo', label='Training Loss')
    plt.plot(epochs, val_loss, 'b', label='Validation Loss')
    plt.title('Training and validation loss')
    plt.legend()
    plt.show()

    plt.plot(epochs, acc, 'r', label='Training accuracy')
    plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
    plt.title('Training and validation accuracy')
    plt.legend(loc=0)
    plt.figure()

    plt.show()