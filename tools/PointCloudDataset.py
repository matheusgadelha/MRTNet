import numpy as np
import glob
import torch.utils.data
import os
from skimage import io, transform

def save_obj_file(pts, filepath):
    f = open(filepath, 'w')
    for v in xrange(pts.shape[0]):
        line = "v {} {} {}\n".format(pts[v, 0], pts[v, 1], pts[v, 2])
        f.write(line)
        if pts[v, :].shape[0] > 3:
            length = np.sqrt(np.sum(np.power(pts[v, 3:6], 2)))
            pts[v, 3:6] /= length
            line = "vn {} {} {}\n".format(pts[v, 3], pts[v, 4], pts[v, 5])
            f.write(line)
    f.close()


def save_torch_pc(path, pc):
    results = pc.cpu().data.numpy()
    results = results.transpose(0, 2, 1)[0, :, :]
    save_obj_file(results, path)

def save_objs(pts, path, start_idx=0):
    filename = "pc_{}.obj"
    npts = pts.shape[1]
    ndims = pts.shape[2]

    for i in xrange(pts.shape[0]):
        save_obj_file(pts[i, :, :].reshape(npts, ndims),
                os.path.join(path, filename.format(str(i+start_idx).zfill(4))))


def save_points(pts, path):
    filename = "pc_{}.npy"
    npts = pts.shape[1]
    ndims = pts.shape[3]

    for i in xrange(pts.shape[0]):
        np.save(path+"/"+filename.format(i),
                pts[i, :, :, :].reshape(npts, ndims))


def point_from_vline(vline):
    sp = vline.split(' ')[1:4]
    pt = []
    for v in sp:
        if np.isnan(float(v)):
            raise ValueError
        pt.append(float(v))
    return np.array(pt)


def read_obj(opath):
    points = []
    with open(opath) as f:
        for line in f:
            if line.startswith('vn'):
                points[-1] = np.append(points[-1], point_from_vline(line))
            else:
                points.append(point_from_vline(line))
    return np.array(points)


def write_image_pc(path, pair):
    #mean = np.array([0.485, 0.456, 0.406]).astype(float)
    #std = np.array([0.229, 0.224, 0.225]).astype(float)

    #img = pair[0].numpy().transpose((1, 2, 0))
    img = pair[0]
    #img = np.clip(img*std + mean, 0.0, 1.0)
    #io.imsave(path+'.png', (img*255).astype('uint8'))
    io.imsave(path+'.png', img)
    save_obj_file(pair[1].transpose(0,1).numpy(), path+'.obj')


class ImageDataset(torch.utils.data.Dataset):

    def __init__(self, image_dir):
        self.img_paths = sorted(glob.glob(image_dir+"/*.png"))

        self.mean = np.array([0.485, 0.456, 0.406]).astype(float)
        self.std = np.array([0.229, 0.224, 0.225]).astype(float)

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img = io.imread(self.img_paths[idx])[:, :, 0:3].astype('float32')
        img /= 255.0
        img = transform.resize(img, (224, 224))
        img = (img - self.mean)/self.std
        img = img.transpose((2, 0, 1))
        img = torch.from_numpy(img).float()

        name = self.img_paths[idx].split("/")[-1].split(".")[-2]

        return (img, io.imread(self.img_paths[idx]), name)


class ImageToPointCloudDataset(torch.utils.data.Dataset):
    
    def __init__(self, image_dir, pc_dir, category="all", train_mode=True):
        self.train_mode = train_mode
        self.category = category

        self.classids= ['02691156', '02828884', '02933112', '02958343', '03001627', 
            '03211117', '03636649', '03691459', '04090263', '04256520', '04379243', 
            '04401088', '04530566']

        self.img_paths = []
        if self.category == "all":
            self.img_paths = sorted(glob.glob(image_dir+"/*/*/rendering/*.png"),
                    key=lambda p: p.split('/')[-3])
        else:
            self.img_paths = sorted(glob.glob(
                    image_dir+"/{}/*/rendering/*.png".format(self.category)),
                    key=lambda p: p.split('/')[-3])
        
        self.mean = np.array([0.485, 0.456, 0.406]).astype(float)
        self.std = np.array([0.229, 0.224, 0.225]).astype(float)

        ntrain_id = int(len(self.img_paths) * 0.8)
        if train_mode:
            self.img_paths = self.img_paths[0:ntrain_id]
        else:
            self.img_paths = self.img_paths[ntrain_id:]
        self.pc_paths = []

        for img in self.img_paths:
            model_signature = img.split('/')[-3]
            pc_path = "{}/{}_0.npy".format(pc_dir, model_signature)
            self.pc_paths.append(pc_path)

    def __len__(self):
        return len(self.img_paths)
    
    def __getitem__(self, idx):
        img = io.imread(self.img_paths[idx])[:, :, 0:3].astype('float32')
        img /= 255.0
        img = transform.resize(img, (224, 224))
        img = (img - self.mean)/self.std
        img = img.transpose((2, 0, 1))
        img = torch.from_numpy(img).float()

        pc = torch.FloatTensor(np.load(self.pc_paths[idx])[:, 0:3].transpose())

        #if self.train_mode:
        #    out = (img, pc)
        #else:
        #    out = (img, pc, self.classids.index(self.img_paths[idx].split('/')[-4]))
        out = (img, pc, self.classids.index(self.img_paths[idx].split('/')[-4]),
                io.imread(self.img_paths[idx]))

        return out


class PointCloudDataset(torch.utils.data.Dataset):
    
    def __init__(self, root_dir):
        self.filepaths = glob.glob(root_dir+"/*.obj")
        self.root_dir = root_dir
    
    def __len__(self):
        return len(self.filepaths)


    def __getitem__(self, idx):
        return torch.FloatTensor(read_obj(self.filepaths[idx])[:, 0:3].transpose())


class PointModelNetDataset(torch.utils.data.Dataset):

    def __init__(self, root_dir, scale_aug=True, rot_aug=False, test_mode=False):
        self.filepaths = sorted(glob.glob(root_dir+"/*.obj_0.kdt.npy"))
        #self.filepaths = sorted(glob.glob(root_dir+"/*.rpt.npy"))
        self.root_dir = root_dir
        self.scale_aug = scale_aug
        self.rot_aug = rot_aug
        self.test_mode = test_mode

        self.class_files = {}

        self.classnames=[
            'airplane',
            'bathtub',
            'bed',
            'bench',
            'bookshelf',
            'bottle',
            'bowl',
            'car',
            'chair',
            'cone',
            'cup',
            'curtain',
            'desk',
            'door',
            'dresser',
            'flower_pot',
            'glass_box',
            'guitar',
            'keyboard',
            'lamp',
            'laptop',
            'mantel',
            'monitor',
            'night_stand',
            'person',
            'piano',
            'plant',
            'radio',
            'range_hood',
            'sink',
            'sofa',
            'stairs',
            'stool',
            'table',
            'tent',
            'toilet',
            'tv_stand',
            'vase',
            'wardrobe',
            'xbox']

        self.class_splits = []

        current_class = '###'
        for f in self.filepaths:
            if current_class in f:
                self.class_splits[-1] += 1
            else:
                for c in self.classnames:
                    if c in f and c != current_class:
                        self.class_splits.append(1)
                        current_class = c

        self.class_splits = np.cumsum(np.array(self.class_splits))


    def __len__(self):
        return len(self.filepaths)


    def __getitem__(self, idx):
        path = self.filepaths[idx]
        class_idx = None
        class_name = path.split('/')[5]
        i = self.classnames.index(class_name)

        pc = np.load(self.filepaths[idx])
        if self.scale_aug:
            scale_factor = np.random.uniform(0.75, 1.5, 3)
            pc[:, 0:3] *= scale_factor

        if self.rot_aug:
            rot_matrix = rotation_matrix(np.array([0, 0, 1]),
                    np.random.choice(np.arange(0, 2*np.pi, np.pi/4)) +
                    (np.random.random_sample()-0.5) + 1e-2)
            pc[:, 0:3] = pc[:, 0:3].dot(rot_matrix)

        dims = pc[:, 0:3].max(axis=0) - pc[:, 0:3].min(axis=0)
        pc[:, 0:3] /= dims.max()

        return (i, torch.FloatTensor(pc.transpose()))


