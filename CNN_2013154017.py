import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import struct
import CNN_logging

def read_mnist(mode):
    if mode == 'training':
        fname_img = 'train-images.idx3-ubyte'
        fname_lbl = 'train-labels.idx1-ubyte'

    elif mode == 'test':
        fname_img = 't10k-images.idx3-ubyte'
        fname_lbl = 't10k-labels.idx1-ubyte'

    else:
        Log.error('mode is incorrect.')

    with open(fname_lbl, 'rb') as file_lbl:
        magic, num = struct.unpack('>II', file_lbl.read(8))
        label = np.fromfile(file_lbl, dtype=np.uint8)

    with open(fname_img, 'rb') as file_img:
        magic, num, rows, cols = struct.unpack('>IIII', file_img.read(16))
        img = np.fromfile(file_img, dtype=np.uint8).reshape(len(label), rows, cols)

    get_img = lambda idx : (label[idx], img[idx])

    for i in range(len(label)):
        yield get_img(i)

def show_mnist(img_array, label='unknown'):
    img = Image.fromarray(img_array, 'L')
    plt.title(label)
    plt.imshow(img)
    plt.show()

class layer_convolution():
    def __init__(self, rows, cols):
        self.rows = rows
        self.cols = cols
        self.padded_img = np.zeros((rows+2, cols+2), dtype=np.uint8)
        self.kernel = None
        self.output_img = np.zeros((rows, cols), dtype=np.uint8)

    # 이미지 입력 후 padding, normalize[0, 1] 수행
    def input_img(self, img):
        print('> {} image has been input'.format(img.shape))
        if img.shape == (self.rows, self.cols):
            self.padded_img[1:self.rows+1, 1:self.cols+1] = img
            self.normalized_img = self.padded_img / 255.0
            print('> {} image has padded into {}'.format(img.shape, self.normalized_img.shape))
        else:
            Log.error('size of input image is incorrect.')

    # 정해진 size의 kernel 생성(Xavier 초깃값 사용)
    def set_kernel(self, size=3):
        self.size_kernel = size
        xavier = np.sqrt(6/self.rows*self.cols)
        self.kernel = np.random.uniform(-xavier, xavier, (size, size))
        print('> {} kernel has been set.'.format(self.kernel.shape))

    def convolution(self, stride=1, bias=0):
        for i in range(self.rows):
            for j in range(self.cols):
                self.output_img[i][j] = np.sum(self.normalized_img[i:i+self.size_kernel, j:j+self.size_kernel]*self.kernel) + bias
        print('> {} image is convolved into {}'.format(self.normalized_img.shape, self.output_img.shape))

        return self.output_img

class layer_Relu():
    def __init__(self):
        pass

    def relu(self, img):
        return np.maximum(0, img)

class layer_pooling():
    def __init__(self, method='max', size=3):
        self.method = method
        self.size = size

    def pooling(self, img, size_window=2):
        show_mnist(img)
        (rows, cols) = img.shape
        ret = np.zeros((int(rows/size_window), int(cols/size_window)), dtype=np.uint8)
        if self.method == 'max':
            for i in range(0, rows, size_window):
                for j in range(0, cols, size_window):
                    sub = np.max(img[i:i+size_window, j:j+size_window])
                    ret[int(i/size_window), int(j/size_window)] = sub
            print('> {} image is pooled into {}'.format(img.shape, ret.shape))
            show_mnist(ret)
            return ret

Log = CNN_logging.set_logging()
iter_data = read_mnist('training')
for (label, img) in iter_data:
    conv1 = layer_convolution(28, 28)
    conv1.input_img(img)
    conv1.set_kernel(size=3)
    passing_img = conv1.convolution(stride=1)

    relu1 = layer_Relu()
    passing_img = relu1.relu(passing_img)

    pooling1 = layer_pooling('max')
    passing_img = pooling1.pooling(passing_img)
    break
