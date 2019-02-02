import torchvision.models as models
from models_src.t3d import DenseNet3D

densenet = models.densenet201
#should probably use higher resolution inputs since our actions take up a smaller space in the image
#also, cant seem to find how to specifiy input dimension,
t3d = DenseNet3D(num_init_features=96, growth_rate=48, block_config=(6, 12, 48, 32), classifier=False)
