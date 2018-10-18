import numpy as np
import cv2
import mxnet as mx
import datetime
from ssh_detector import SSHDetector


count = 0
Path = '/home/infinova/chengw/FDDB/FDDB-folds/'
f = open('/home/infinova/chengw/FDDB/test/fddb_dets.txt', 'wt')
ctx = mx.cpu()

scales = [400, 800]
target_size = scales[0]
max_size = scales[1]
detector = SSHDetector('./model/e2e', 0)

for Name in open('/home/infinova/chengw/FDDB/test/fddb_img_list.txt'):
    im_scale = 1.0
    Image_Path = Path + Name[:-1] + '.jpg'
    img = cv2.imread(Image_Path)
    heigh = img.shape[0]
    width = img.shape[1]

    im_shape = img.shape
    im_size_min = np.min(im_shape[0:2])
    im_size_max = np.max(im_shape[0:2])
    if im_size_min > target_size or im_size_max > max_size:
        im_scale = float(target_size) / float(im_size_min)
        # prevent bigger axis from being more than max_size:
        if np.round(im_scale * im_size_max) > max_size:
            im_scale = float(max_size) / float(im_size_max)
        img = cv2.resize(img, None, None, fx=im_scale, fy=im_scale)
        print('resize to', img.shape)
    timea = datetime.datetime.now()
    faces = detector.detect(img, threshold=0.1)
    timeb = datetime.datetime.now()
    f.write('{:s}\n'.format(Name[:-1]))
    f.write('{:.1f}\n'.format(faces.shape[0]))
    for num in range(faces.shape[0]):
        bbox = faces[num, :]
        xmin = bbox[0] / im_scale
        ymin = bbox[1] / im_scale
        xmax = bbox[2] / im_scale
        ymax = bbox[3] / im_scale
        score = bbox[4]
        f.write('{:.3f} {:.3f} {:.3f} {:.3f} {:.10f}\n'.
                format(xmin, ymin, (xmax - xmin + 1), (ymax - ymin + 1), score))
    count += 1
    print('%d/2845' % count)
