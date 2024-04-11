# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
import tempfile
import warnings
import os
import mmcv
import numpy as np
import torch
from mmcv.engine import collect_results_cpu, collect_results_gpu
from mmcv.image import tensor2imgs
from mmcv.runner import get_dist_info
import cv2
import numpy as np
from PIL import Image as PILImage
def mkdir(path):
    sub_dir = os.path.dirname(path)
    if not os.path.exists(sub_dir):
        os.makedirs(sub_dir)

def  get_pseudo_color_map(pred):
    pred_mask = PILImage.fromarray(pred.astype(np.uint8), mode='P')
    color_map = get_color_map_list(256)
#     color_map = get_cityscapes_colors()
    pred_mask.putpalette(color_map)
    return pred_mask

def get_cityscapes_colors():
    """ Returns the color map for visualizing the segmentation mask.
    Args:
        num_cls: Number of classes
    Returns:
        The color map
    """
    num_cls = 20
    colors = [0] * (num_cls * 3)
    colors[0:3] = (128, 64, 128)       # 0: 'road' 
    colors[3:6] = (244, 35,232)        # 1 'sidewalk'
    colors[6:9] = (70, 70, 70)         # 2''building'
    colors[9:12] = (102,102,156)       # 3 wall
    colors[12:15] =  (190,153,153)     # 4 fence
    colors[15:18] = (153,153,153)      # 5 pole
    colors[18:21] = (250,170, 30)      # 6 'traffic light'
    colors[21:24] = (220,220, 0)       # 7 'traffic sign'
    colors[24:27] = (107,142, 35)      # 8 'vegetation'
    colors[27:30] = (152,251,152)      # 9 'terrain'
    colors[30:33] = ( 70,130,180)      # 10 sky
    colors[33:36] = (220, 20, 60)      # 11 person
    colors[36:39] = (255, 0, 0)        # 12 rider
    colors[39:42] = (0, 0, 142)        # 13 car
    colors[42:45] = (0, 0, 70)         # 14 truck
    colors[45:48] = (0, 60,100)        # 15 bus
    colors[48:51] = (0, 80,100)        # 16 train
    colors[51:54] = (0, 0,230)         # 17 'motorcycle'
    colors[54:57] = (119, 11, 32)      # 18 'bicycle'
    colors[57:60] = (105, 105, 105)
    return colors

def get_color_map_list(num_classes):
    """
    Returns the color map for visualizing the segmentation mask,
    which can support arbitrary number of classes.

    Args:
        num_classes (int): Number of classes.

    Returns:
        (list). The color map.
    """

    num_classes += 1
    color_map = num_classes * [0, 0, 0]
    for i in range(0, num_classes):
        j = 0
        lab = i
        while lab:
            color_map[i * 3] |= (((lab >> 0) & 1) << (7 - j))
            color_map[i * 3 + 1] |= (((lab >> 1) & 1) << (7 - j))
            color_map[i * 3 + 2] |= (((lab >> 2) & 1) << (7 - j))
            j += 1
            lab >>= 3
#     color_map = [color_map[i:i + 3] for i in range(0, len(color_map), 3)]
    color_map = color_map[3:]
    return color_map

def np2tmp(array, temp_file_name=None, tmpdir=None):
    """Save ndarray to local numpy file.

    Args:
        array (ndarray): Ndarray to save.
        temp_file_name (str): Numpy file name. If 'temp_file_name=None', this
            function will generate a file name with tempfile.NamedTemporaryFile
            to save ndarray. Default: None.
        tmpdir (str): Temporary directory to save Ndarray files. Default: None.
    Returns:
        str: The numpy file name.
    """

    if temp_file_name is None:
        temp_file_name = tempfile.NamedTemporaryFile(
            suffix='.npy', delete=False, dir=tmpdir).name
    np.save(temp_file_name, array)
    return temp_file_name


def single_gpu_test(model,
                    data_loader,
                    show=False,
                    out_dir=None,
                    efficient_test=False,
                    opacity=0.5,
                    pre_eval=False,
                    format_only=False,
                    format_args={}):
    """Test with single GPU by progressive mode.

    Args:
        model (nn.Module): Model to be tested.
        data_loader (utils.data.Dataloader): Pytorch data loader.
        show (bool): Whether show results during inference. Default: False.
        out_dir (str, optional): If specified, the results will be dumped into
            the directory to save output results.
        efficient_test (bool): Whether save the results as local numpy files to
            save CPU memory during evaluation. Mutually exclusive with
            pre_eval and format_results. Default: False.
        opacity(float): Opacity of painted segmentation map.
            Default 0.5.
            Must be in (0, 1] range.
        pre_eval (bool): Use dataset.pre_eval() function to generate
            pre_results for metric evaluation. Mutually exclusive with
            efficient_test and format_results. Default: False.
        format_only (bool): Only format result for results commit.
            Mutually exclusive with pre_eval and efficient_test.
            Default: False.
        format_args (dict): The args for format_results. Default: {}.
    Returns:
        list: list of evaluation pre-results or list of save file names.
    """
    if efficient_test:
        warnings.warn(
            'DeprecationWarning: ``efficient_test`` will be deprecated, the '
            'evaluation is CPU memory friendly with pre_eval=True')
        mmcv.mkdir_or_exist('.efficient_test')
    # when none of them is set true, return segmentation results as
    # a list of np.array.
    assert [efficient_test, pre_eval, format_only].count(True) <= 1, \
        '``efficient_test``, ``pre_eval`` and ``format_only`` are mutually ' \
        'exclusive, only one of them could be true .'

    model.eval()
    results = []
    dataset = data_loader.dataset
    prog_bar = mmcv.ProgressBar(len(dataset))
    # The pipeline about how the data_loader retrieval samples from dataset:
    # sampler -> batch_sampler -> indices
    # The indices are passed to dataset_fetcher to get data from dataset.
    # data_fetcher -> collate_fn(dataset[index]) -> data_sample
    # we use batch_sampler to get correct data idx
    loader_indices = data_loader.batch_sampler

    for batch_indices, data in zip(loader_indices, data_loader):
        with torch.no_grad():
            result = model(return_loss=False, **data)

        if show or out_dir:
            img_tensor = data['img'][0]
            img_metas = data['img_metas'][0].data[0]
            imgs = tensor2imgs(img_tensor, **img_metas[0]['img_norm_cfg'])
            assert len(imgs) == len(img_metas)

            for img, img_meta in zip(imgs, img_metas):
                h, w, _ = img_meta['img_shape']
                img_show = img[:h, :w, :]

                ori_h, ori_w = img_meta['ori_shape'][:-1]
                img_show = mmcv.imresize(img_show, (ori_w, ori_h))

                if out_dir:
                    out_file = osp.join(out_dir, img_meta['ori_filename'])
                else:
                    out_file = None

                model.module.show_result(
                    img_show,
                    result,
                    palette=dataset.PALETTE,
                    show=show,
                    out_file=out_file,
                    opacity=opacity)

        if efficient_test:
            result = [np2tmp(_, tmpdir='.efficient_test') for _ in result]

        if format_only:
            result = dataset.format_results(
                result, indices=batch_indices, **format_args)
        if pre_eval:
            # TODO: adapt samples_per_gpu > 1.
            # only samples_per_gpu=1 valid now
            result = dataset.pre_eval(result, indices=batch_indices)
            results.extend(result)
        else:
            results.extend(result)

        batch_size = len(result)
        for _ in range(batch_size):
            prog_bar.update()

    return results


# def multi_gpu_test(model,
#                    data_loader,
#                    tmpdir=None,
#                    gpu_collect=False,
#                    efficient_test=False,
#                    pre_eval=False,
#                    format_only=False,
#                    format_args={}):
#     """Test model with multiple gpus by progressive mode.

#     This method tests model with multiple gpus and collects the results
#     under two different modes: gpu and cpu modes. By setting 'gpu_collect=True'
#     it encodes results to gpu tensors and use gpu communication for results
#     collection. On cpu mode it saves the results on different gpus to 'tmpdir'
#     and collects them by the rank 0 worker.

#     Args:
#         model (nn.Module): Model to be tested.
#         data_loader (utils.data.Dataloader): Pytorch data loader.
#         tmpdir (str): Path of directory to save the temporary results from
#             different gpus under cpu mode. The same path is used for efficient
#             test. Default: None.
#         gpu_collect (bool): Option to use either gpu or cpu to collect results.
#             Default: False.
#         efficient_test (bool): Whether save the results as local numpy files to
#             save CPU memory during evaluation. Mutually exclusive with
#             pre_eval and format_results. Default: False.
#         pre_eval (bool): Use dataset.pre_eval() function to generate
#             pre_results for metric evaluation. Mutually exclusive with
#             efficient_test and format_results. Default: False.
#         format_only (bool): Only format result for results commit.
#             Mutually exclusive with pre_eval and efficient_test.
#             Default: False.
#         format_args (dict): The args for format_results. Default: {}.

#     Returns:
#         list: list of evaluation pre-results or list of save file names.
#     """
#     if efficient_test:
#         warnings.warn(
#             'DeprecationWarning: ``efficient_test`` will be deprecated, the '
#             'evaluation is CPU memory friendly with pre_eval=True')
#         mmcv.mkdir_or_exist('.efficient_test')
#     # when none of them is set true, return segmentation results as
#     # a list of np.array.
#     assert [efficient_test, pre_eval, format_only].count(True) <= 1, \
#         '``efficient_test``, ``pre_eval`` and ``format_only`` are mutually ' \
#         'exclusive, only one of them could be true .'

#     model.eval()
#     results = []
#     dataset = data_loader.dataset
#     # The pipeline about how the data_loader retrieval samples from dataset:
#     # sampler -> batch_sampler -> indices
#     # The indices are passed to dataset_fetcher to get data from dataset.
#     # data_fetcher -> collate_fn(dataset[index]) -> data_sample
#     # we use batch_sampler to get correct data idx

#     # batch_sampler based on DistributedSampler, the indices only point to data
#     # samples of related machine.
#     loader_indices = data_loader.batch_sampler

#     rank, world_size = get_dist_info()
#     if rank == 0:
#         prog_bar = mmcv.ProgressBar(len(dataset))

#     for batch_indices, data in zip(loader_indices, data_loader):
#         with torch.no_grad():
#             result = model(return_loss=False, rescale=True, **data)

#         if efficient_test:
#             result = [np2tmp(_, tmpdir='.efficient_test') for _ in result]

#         if format_only:
#             result = dataset.format_results(
#                 result, indices=batch_indices, **format_args)
#         if pre_eval:
#             # TODO: adapt samples_per_gpu > 1.
#             # only samples_per_gpu=1 valid now
#             result = dataset.pre_eval(result, indices=batch_indices)

#         results.extend(result)

#         if rank == 0:
#             batch_size = len(result) * world_size
#             for _ in range(batch_size):
#                 prog_bar.update()

#     # collect results from all ranks
#     if gpu_collect:
#         results = collect_results_gpu(results, len(dataset))
#     else:
#         results = collect_results_cpu(results, len(dataset), tmpdir)
#     return results


def multi_gpu_test(model,
                   data_loader,
                   tmpdir=None,
                   gpu_collect=False,
                   efficient_test=False,
                   pre_eval=False,
                   format_only=False,
                   format_args={}):
    """Test model with multiple gpus by progressive mode.

    This method tests model with multiple gpus and collects the results
    under two different modes: gpu and cpu modes. By setting 'gpu_collect=True'
    it encodes results to gpu tensors and use gpu communication for results
    collection. On cpu mode it saves the results on different gpus to 'tmpdir'
    and collects them by the rank 0 worker.

    Args:
        model (nn.Module): Model to be tested.
        data_loader (utils.data.Dataloader): Pytorch data loader.
        tmpdir (str): Path of directory to save the temporary results from
            different gpus under cpu mode. The same path is used for efficient
            test. Default: None.
        gpu_collect (bool): Option to use either gpu or cpu to collect results.
            Default: False.
        efficient_test (bool): Whether save the results as local numpy files to
            save CPU memory during evaluation. Mutually exclusive with
            pre_eval and format_results. Default: False.
        pre_eval (bool): Use dataset.pre_eval() function to generate
            pre_results for metric evaluation. Mutually exclusive with
            efficient_test and format_results. Default: False.
        format_only (bool): Only format result for results commit.
            Mutually exclusive with pre_eval and efficient_test.
            Default: False.
        format_args (dict): The args for format_results. Default: {}.

    Returns:
        list: list of evaluation pre-results or list of save file names.
    """
    if efficient_test:
        warnings.warn(
            'DeprecationWarning: ``efficient_test`` will be deprecated, the '
            'evaluation is CPU memory friendly with pre_eval=True')
        mmcv.mkdir_or_exist('.efficient_test')
    # when none of them is set true, return segmentation results as
    # a list of np.array.
    assert [efficient_test, pre_eval, format_only].count(True) <= 1, \
        '``efficient_test``, ``pre_eval`` and ``format_only`` are mutually ' \
        'exclusive, only one of them could be true .'

    model.eval()
    results = []
    dataset = data_loader.dataset
    # The pipeline about how the data_loader retrieval samples from dataset:
    # sampler -> batch_sampler -> indices
    # The indices are passed to dataset_fetcher to get data from dataset.
    # data_fetcher -> collate_fn(dataset[index]) -> data_sample
    # we use batch_sampler to get correct data idx

    # batch_sampler based on DistributedSampler, the indices only point to data
    # samples of related machine.
    loader_indices = data_loader.batch_sampler

    rank, world_size = get_dist_info()
    if rank == 0:
        prog_bar = mmcv.ProgressBar(len(dataset))

    for batch_indices, data in zip(loader_indices, data_loader):
        with torch.no_grad():
            result = model(return_loss=False, rescale=True, **data)
            
            img_id = data['img_metas'][0].data[0][0]['ori_filename']
#             if not os.path.exists(os.path.join('results', 'val', img_id.split('_')[0], img_id[len(img_id.split('_')[0])+1:-3]+'png')):
#                 result = model(return_loss=False, rescale=True, **data)
#                 pred_mask = get_pseudo_color_map(result[0])
#                 pred_saved_path = os.path.join('results', 'val', img_id.split('_')[0], img_id[len(img_id.split('_')[0])+1:-3]+'png')
            if not os.path.exists(os.path.join('results', 'val', img_id[:-3]+'png')):
                result = model(return_loss=False, rescale=True, **data)
                pred_mask = get_pseudo_color_map(result[0])
                pred_saved_path = os.path.join('results', 'val', img_id[:-3]+'png')
                
                mkdir(pred_saved_path)
                pred_mask.save(pred_saved_path)             

#         if efficient_test:
#             result = [np2tmp(_, tmpdir='.efficient_test') for _ in result]

#         if format_only:
#             result = dataset.format_results(
#                 result, indices=batch_indices, **format_args)
#         if pre_eval:
#             # TODO: adapt samples_per_gpu > 1.
#             # only samples_per_gpu=1 valid now
#             result = dataset.pre_eval(result, indices=batch_indices)

#         results.extend(result)

        if rank == 0:
            batch_size = len(result) * world_size
            for _ in range(batch_size):
                prog_bar.update()

    # collect results from all ranks
    if gpu_collect:
        results = collect_results_gpu(results, len(dataset))
    else:
        results = collect_results_cpu(results, len(dataset), tmpdir)
    return results