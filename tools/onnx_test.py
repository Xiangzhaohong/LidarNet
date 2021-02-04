import argparse

import numpy as np
import torch

from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.datasets import build_dataloader
from pcdet.models import build_network
from pcdet.utils import common_utils
from pcdet.models import load_data_to_gpu
# Some standard imports
import io

from torch import nn
import torch.utils.model_zoo as model_zoo
import torch.onnx

import torch.nn as nn
import torch.nn.init as init
import time
# import tensorrt


# def parse_config():
#     parser = argparse.ArgumentParser(description='arg parser')
#     parser.add_argument('--cfg_file', type=str, default='cfgs/robosense_models/robosense_pointpillar.yaml', help='specify the config for training')
#     parser.add_argument('--batch_size', type=int, default=1, required=False, help='batch size for training')
#     parser.add_argument('--workers', type=int, default=4, help='number of workers for dataloader')
#     parser.add_argument('--ckpt', type=str, default='/home/syang/Projects/OpenPCDet/output/robosense_models/robosense_pointpillar/BResampl_LR001/ckpt/checkpoint_epoch_25.pth', help='checkpoint to start from')
#     args = parser.parse_args()
#
#     cfg_from_yaml_file(args.cfg_file, cfg)
#     np.random.seed(1024)
#
#     return args, cfg
#
#
# def main():
#     args, cfg = parse_config()
#     logger = common_utils.create_logger('onnx_test.text', rank=0)
#
#     test_set, test_loader, sampler = build_dataloader(
#         dataset_cfg=cfg.DATA_CONFIG,
#         class_names=cfg.CLASS_NAMES,
#         batch_size=args.batch_size,
#         dist=False, workers=args.workers, training=False, logger=logger
#     )
#
#     model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=test_set)
#     with torch.no_grad():
#         model.load_params_from_file(filename=args.ckpt, logger=logger)
#         model.cuda()
#         model.eval()
#         for i, batch_dict in enumerate(test_loader):
#
#             load_data_to_gpu(batch_dict)
#             with torch.no_grad():
#                 pred_dicts, ret_dict = model(batch_dict)


class SuperResolutionNet(nn.Module):
    def __init__(self, upscale_factor, inplace=False):
        super(SuperResolutionNet, self).__init__()

        self.relu = nn.ReLU(inplace=inplace)
        # self.conv1 = nn.Conv2d(1, 64, (5, 5), (1, 1), (2, 2))
        self.conv1 = nn.Conv2d(1, 64, (5, 5))
        self.conv2 = nn.Conv2d(64, 64, (3, 3), (1, 1), (1, 1))
        self.conv3 = nn.Conv2d(64, 32, (3, 3), (1, 1), (1, 1))
        self.conv4 = nn.Conv2d(32, upscale_factor ** 2, (3, 3), (1, 1), (1, 1))
        self.pixel_shuffle = nn.PixelShuffle(upscale_factor)

        self._initialize_weights()

    def forward(self, x, y):
        # x = dict['x']
        # y = dict['y']
        # x = self.relu(self.conv1(x))
        # x = self.relu(self.conv2(x))
        # x = self.relu(self.conv3(x))
        # x = self.pixel_shuffle(self.conv4(x))

        x = self.conv1(x)
        # x = self.conv2(x)
        # x = self.conv3(x)

        y = self.conv1(y)
        # y = self.conv2(y)
        # y = self.conv3(y)

        temp = x + y
        output_dumpy = y + y
        # dict_output = {}
        # dict_output['out_1'] = temp
        # dict_output['out_2'] = output_dumpy
        return temp

    def _initialize_weights(self):
        init.orthogonal_(self.conv1.weight, init.calculate_gain('relu'))
        init.orthogonal_(self.conv2.weight, init.calculate_gain('relu'))
        init.orthogonal_(self.conv3.weight, init.calculate_gain('relu'))
        init.orthogonal_(self.conv4.weight)


def test_pytorch_sample():
    torch_model = SuperResolutionNet(upscale_factor=3)
    model_url = 'https://s3.amazonaws.com/pytorch/test_data/export/superres_epoch100-44c6958e.pth'
    batch_size = 1  # just a random number
    # Initialize model with the pretrained weights
    map_location = lambda storage, loc: storage
    if torch.cuda.is_available():
        map_location = None
    torch_model.load_state_dict(model_zoo.load_url(model_url, map_location=map_location))

    # set the model to inference mode
    torch_model.eval()
    # Input to the model
    x = torch.randn(batch_size, 1, 224, 224, dtype=torch.float32)
    y = torch.randn(batch_size, 1, 224, 224, dtype=torch.float32)
    dict = {'x':x, 'y':y}
    # torch_out = torch_model(dict)
    torch_out = torch_model(x, y)

    # Export the model
    torch.onnx.export(torch_model,  # model being run
                      (x, y),  # model input (or a tuple for multiple inputs)
                      "super_resolution.onnx",  # where to save the model (can be a file or file-like object)
                      # export_params=True,  # store the trained parameter weights inside the model file
                      opset_version=11,  # the ONNX version to export the model to
                      verbose=True,
                      # do_constant_folding=True,  # whether to execute constant folding for optimization
                      input_names=['input', 'dumpy_y']  # the model's input names
                     #,output_names=['output', 'dumpy_out']  # the model's output names
                      # ,dynamic_axes={'input': {0: 'batch_size'},  # variable lenght axes
                      #               'dumpy_y': {0: 'batch_size'}
                      #                # ,'output': {0: 'batch_size'}
                      #                # , 'dumpy_out': {0: 'batch_size'}
                      #                }
                    # ,example_outputs=torch_out
                      )

    # import onnx
    # onnx_model = onnx.load("super_resolution.onnx")
    # onnx.checker.check_model(onnx_model)
    # onnx.helper.printable_graph(onnx_model.graph)
    #
    # import onnxruntime
    # ort_session = onnxruntime.InferenceSession("super_resolution.onnx")
    # def to_numpy(tensor):
    #     return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()
    # # compute ONNX Runtime output prediction
    # x = torch.randn(1, 1, 224, 224, dtype=torch.float32)
    # y = torch.randn(1, 1, 224, 224, dtype=torch.float32)
    # dict = {'x': x, 'y': y}
    #
    # start_torch = time.time()
    # for i in range(1):
    #     # torch_out, dumpy_out = torch_model(dict)
    #     torch_out, dumpy_out = torch_model(x,y)
    # end_torch = time.time()
    # print('model inference time is : ', (end_torch - start_torch)/1000)
    #
    # ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(x),
    #               ort_session.get_inputs()[1].name: to_numpy(y)}
    #
    # start_run = time.time()
    # for i in range(1):
    #     ort_outs, ort_dumpy_outputs = ort_session.run(None, ort_inputs)
    # end_run = time.time()
    # print('onnx_runtime model inference time is : ', (end_run - start_run) / 1000)
    #
    # # compare ONNX Runtime and PyTorch results
    # np.testing.assert_allclose(to_numpy(torch_out), ort_outs, rtol=1e-03, atol=1e-05)
    # np.testing.assert_allclose(to_numpy(dumpy_out), ort_dumpy_outputs, rtol=1e-03, atol=1e-05)
    #
    # print("Exported model has been tested with ONNXRuntime, and the result looks good!")

    print('end for debug!')


# def test_onnx_tensorrt():
#     import onnx
#     import tools.onnx_tensorrt.backend as backend
#     import numpy as np
#
#     model = onnx.load("super_resolution.onnx")
#     engine = backend.prepare(model, device='CUDA:0')
#
#     # input_data = np.random.random(size=(32, 3, 224, 224)).astype(np.float32)
#     x = np.random.random(size=(1, 1, 224, 224)).astype(np.float32)
#     y = np.random.random(size=(1, 1, 224, 224)).astype(np.float32)
#     # input_data = {'x': x, 'y': y}
#     input_data = [x,y]
#
#     output_data = engine.run(input_data)[0]
#     print(output_data)
#     print(output_data.shape)
#
#     print('test test_onnx_tensorrt end!')


# def test_my_onnx_tensorrt():
#     import onnx
#     import numpy as np
#     import tensorrt as trt
#     TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
#     builder = trt.Builder(TRT_LOGGER)
#     network = builder.create_network()
#     parser = trt.OnnxParser(network, TRT_LOGGER)
#
#     with open("super_resolution.onnx", 'rb') as model:
#         parser.parse(model.read())
#
#     builder.max_batch_size = 1
#     builder.max_workspace_size = 1 << 20  # This determines the amount of memory available to the builder when building an optimized engine and should generally be set as high as possible.
#
#     config = builder.create_builder_config()
#     engine = builder.build_cuda_engine(network, config)
#     # Do inference here.

    print('test_my_onnx_tensorrt end!')

if __name__ == '__main__':
    # main()
    print(torch.cuda.device_count())
    print('just test!')
    test_pytorch_sample()
    # test_onnx_tensorrt()
    #test_my_onnx_tensorrt()
